from easydict import EasyDict

from configs.train_config import cfg as train_cfg

cfg = EasyDict()

pickles = ['word2int_last.pickle', 'words_last.pickle', 'word2int_bucket_last.pickle', 'words_bucket_last.pickle'] \
    if train_cfg.use_bigrams else ['word2int_.pickle', 'words_.pickle', 'word2int_bucket.pickle', 'words_bucket.pickle']

cfg.dataset_path = '../data/'
cfg.preprocessed_dataset_path = cfg.dataset_path + 'preprocessed_data/'
cfg.word2int_path = cfg.preprocessed_dataset_path + pickles[0]
cfg.words_path = cfg.preprocessed_dataset_path + pickles[1]

cfg.word2int_bucket_path = cfg.preprocessed_dataset_path + pickles[2]
cfg.words_bucket_path = cfg.preprocessed_dataset_path + pickles[3]

cfg.load_preprocessed_data = True
cfg.load_arrays = True

cfg.numbers_alias = {'0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                     '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'}

cfg.min_n = 3
cfg.max_n = 6

cfg.max_vocab_size = 30000000
cfg.bucket_size = 2000000
cfg.thr_for_pruning = int(round(0.75 * cfg.max_vocab_size))
cfg.min_threshold = 1
cfg.min_count = 1

cfg.fin_vocab_size = 37670
cfg.fin_bucket_size = 190050
cfg.fin_bucket_size_with_bigrams = 908609
