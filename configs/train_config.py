from easydict import EasyDict

cfg = EasyDict()

cfg.batch_size = 128
cfg.lr = 0.5  # 1e-3
cfg.weight_decay = 1e-4
cfg.epochs = 5
cfg.use_bigrams = True
cfg.use_multiprocessing = False
cfg.num_processes = 20

cfg.log_metrics = False
cfg.experiment_name = 'Fasttext bigrams'  # 'Fasttext no-bigrams'

cfg.evaluate_on_train_set = True
cfg.evaluate_before_training = True
cfg.eval_plots_dir = f'../saved_files/plots/{cfg.experiment_name}'
cfg.plot_conf_matrices = False

cfg.load_saved_model = True
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 1 if cfg.experiment_name == 'Fasttext bigrams' else 2
cfg.save_model = False
cfg.epochs_saving_freq = 1
