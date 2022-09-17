from easydict import EasyDict


cfg = EasyDict()

cfg.dataset_path = '../data/'
cfg.preprocessed_dataset_path = cfg.dataset_path + 'preprocessed_data/'
cfg.word2int_path = cfg.preprocessed_dataset_path + 'word2int_.pickle'
cfg.words_path = cfg.preprocessed_dataset_path + 'words_.pickle'

cfg.word2int_bucket_path = cfg.preprocessed_dataset_path + 'word2int_bucket.pickle'
cfg.words_bucket_path = cfg.preprocessed_dataset_path + 'words_bucket.pickle'

cfg.load_preprocessed_data = True
cfg.load_arrays = False

cfg.numbers_alias = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                     '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

cfg.min_n = 3
cfg.max_n = 6

cfg.max_vocab_size = 30000000
cfg.bucket_size = 2000000
cfg.thr_for_pruning = int(round(0.75 * cfg.max_vocab_size))
cfg.min_threshold = 1
cfg.min_count = 2

cfg.fin_vocab_size = 37670
cfg.fin_bucket_size = 190050
