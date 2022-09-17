import torch.multiprocessing as mp
import torch
import numpy as np
import os
import time

from dataloader.dataloader import get_dataloaders
from models.FastText import get_model
from utils.logging import Logger
from utils.visualization import plot_confusion_matrix


class Trainer(object):
    def __init__(self, cfg):
        """
        Class for initializing and performing training procedure.
        :param cfg: train config
        """
        self.cfg = cfg
        self.dl_train, self.dl_test = get_dataloaders()
        self.model = get_model()
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.logger = Logger(self.cfg)

    @staticmethod
    def get_criterion():
        """
        Gets criterion.
        :return: criterion
        """
        criterion = torch.nn.NLLLoss()  # nn.CrossEntropyLoss()  #
        return criterion

    def get_optimizer(self):
        """
        Gets optimizer.
        :return: optimizer
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, momentum=0.99, nesterov=True,
                                    weight_decay=self.cfg.weight_decay)
        # optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.cfg.lr,
        #                                'weight_decay': self.cfg.weight_decay}])
        return optimizer

    def restore_model(self):
        """
        Restores saved model.
        """
        if self.cfg.load_saved_model:
            print(f'Trying to load checkpoint from epoch {self.cfg.epoch_to_load}...')
            try:
                checkpoint = torch.load(self.cfg.checkpoints_dir + f'/checkpoint_{self.cfg.epoch_to_load}.pth')
                load_state_dict = checkpoint['model']
                self.model.load_state_dict(load_state_dict)
                self.start_epoch = checkpoint['epoch'] + 1
                self.global_step = checkpoint['global_step'] + 1
                self.optimizer.load_state_dict(checkpoint['opt'])
                print(f'Loaded checkpoint from epoch {self.cfg.epoch_to_load}.')
            except FileNotFoundError:
                print('Checkpoint not found')

    def save_model(self):
        """
        Saves model.
        """
        if self.cfg.save_model and self.epoch % self.cfg.epochs_saving_freq == 0:
            print('Saving current model...')
            state = {
                'model': self.model.state_dict(),
                'epoch': self.epoch,
                'global_step': self.global_step,
                'opt': self.optimizer.state_dict()
            }
            if not os.path.exists(self.cfg.checkpoints_dir):
                os.makedirs(self.cfg.checkpoints_dir)

            path_to_save = os.path.join(self.cfg.checkpoints_dir, f'checkpoint_{self.epoch}.pth')
            torch.save(state, path_to_save)
            print(f'Saved model to {path_to_save}.')

    def evaluate(self, dl, set_type):
        """
        Evaluates model performance. Calculates and logs model accuracy on given data set.
        :param dl: train or test dataloader
        :param set_type: 'train' or 'test' data type
        """
        if not os.path.exists(self.cfg.eval_plots_dir):
            os.makedirs(self.cfg.eval_plots_dir)
        all_predictions, all_labels, cross_entropy_losses = [], [], []

        self.model.eval()
        with torch.no_grad():
            print(f'Evaluating on {set_type} data...')
            eval_start_time = time.time()

            correct_predictions, total_predictions = 0, 0
            dl_len = len(dl)
            for i, batch in enumerate(dl):
                input_vector, labels, lens_ = batch[0], batch[1], batch[2]  # .cuda()

                if i % 500 == 0:
                    print(f'iter: {i}/{dl_len}')

                out = self.model(input_vector, lens_)
                _, predictions = torch.max(out.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += torch.sum(predictions == labels)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                cross_entropy_loss = self.criterion(out, labels)
                cross_entropy_losses.append(cross_entropy_loss.item())

            accuracy = 100 * correct_predictions.item() / total_predictions
            mean_loss = np.mean(cross_entropy_losses)
            print(f'Accuracy on {set_type} data: {accuracy} %, {set_type} error: {100 - accuracy} %, loss: {mean_loss}')

            if self.cfg.plot_conf_matrices:
                plot_confusion_matrix(self.cfg, all_labels, all_predictions, self.epoch, set_type)

            self.logger.log_metrics(names=[f'eval/{set_type}/accuracy', f'eval/{set_type}/loss'],
                                    metrics=[accuracy, mean_loss], step=self.epoch)
            print(f'Evaluating time: {round((time.time() - eval_start_time) / 60, 3)} min')
        self.model.train()

    def make_training_step(self, batch):
        """
        Makes single training step.
        :param batch: current batch containing input vector and it`s label
        :return: loss on current batch
        """
        input_vector, label, lens_ = batch[0], batch[1], batch[2]  # .cuda()
        self.optimizer.zero_grad()
        out = self.model(input_vector, lens_)
        loss = self.criterion(out, label)
        assert not torch.isnan(loss)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def training_loop(self):
        # start training
        print(f'Starting training...')
        iter_num = len(self.dl_train)
        for epoch in range(self.start_epoch, self.cfg.epochs):
            epoch_start_time = time.time()
            self.epoch = epoch
            print(f'Epoch: {self.epoch}/{self.cfg.epochs}')

            losses = []
            for iter_, batch in enumerate(self.dl_train):

                loss = self.make_training_step(batch)
                self.logger.log_metrics(names=['train/loss'], metrics=[loss], step=self.global_step)

                if loss is not None:
                    losses.append(loss)
                self.global_step += 1

                if iter_ % 50 == 0:
                    mean_loss = np.mean(losses[-50:]) if len(losses) > 50 else np.mean(losses)
                    print(f'iter: {iter_}/{iter_num}, loss: {mean_loss}')

            self.logger.log_metrics(names=['train/mean_loss_per_epoch'], metrics=[np.mean(losses)], step=self.epoch)

            # save model
            self.save_model()

            # evaluate on train and test data
            if self.cfg.evaluate_on_train_set:
                self.evaluate(self.dl_train, set_type='train')
            self.evaluate(self.dl_test, set_type='test')

            print(f'Epoch total time: {round((time.time() - epoch_start_time) / 60, 3)} min')

    def train(self):
        """
        Runs training procedure.
        """
        total_training_start_time = time.time()
        self.start_epoch, self.epoch, self.global_step = 0, -1, 0

        # restore model if necessary
        self.restore_model()

        # evaluate on train and test data before training
        if self.cfg.evaluate_before_training:
            if self.cfg.evaluate_on_train_set:
                self.evaluate(self.dl_train, set_type='train')
            self.evaluate(self.dl_test, set_type='test')

        if self.cfg.use_multiprocessing:
            self.num_processes = self.cfg.num_processes
            self.model.share_memory()

            processes = []
            for rank in range(self.num_processes):
                p = mp.Process(target=self.training_loop)
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        else:
            self.training_loop()

        print(f'Training time: {round((time.time() - total_training_start_time) / 60, 3)} min')
