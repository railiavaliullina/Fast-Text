from easydict import EasyDict

cfg = EasyDict()

cfg.batch_size = 128
cfg.lr = 1e-3
cfg.weight_decay = 1e-4
cfg.epochs = 5

cfg.log_metrics = False
cfg.experiment_name = '1_hidden_layer_128_dim'

cfg.evaluate_on_train_set = False
cfg.evaluate_before_training = True
cfg.eval_plots_dir = f'../saved_files/plots/{cfg.experiment_name}'
cfg.plot_conf_matrices = False

cfg.load_saved_model = False
cfg.checkpoints_dir = f'../saved_files/checkpoints/{cfg.experiment_name}'
cfg.epoch_to_load = 9
cfg.save_model = False
cfg.epochs_saving_freq = 1
