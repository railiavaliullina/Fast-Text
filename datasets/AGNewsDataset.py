import torch
import time
import numpy as np
import spacy
import re

from utils.dataframes_handler import read_file, save_file, save_list_as_pickle, read_list


class AGNewsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg, train_cfg, dataset_type):
        """
        Class for reading, preprocessing, encoding and sampling data.
        :param cfg: dataset config
        :param dataset_type: 'train' or 'test' data type
        """
        self.cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.dataset_type = dataset_type
        self.words_ = list(np.ones(self.cfg.max_vocab_size) * -1)
        self.word2int_ = list(np.ones(self.cfg.max_vocab_size) * -1)
        self.last_index_in_words_ = 0

        self.words_bucket = list(np.ones(self.cfg.bucket_size) * -1)
        self.word2int_bucket = list(np.ones(self.cfg.bucket_size) * -1)
        self.last_index_in_bucket = 0

        # read data
        self.read_data()

        # load preprocessed data if possible, else preprocess
        self.preprocess_texts()
        self.texts_num = self.__len__()

        # get words_, word2int_ arrays
        self.get_arrays_()

    def read_data(self):
        """
        Reads source data.
        """
        csv = read_file(self.cfg.dataset_path + f'{self.dataset_type}.csv')
        self.labels = csv['Class Index'].to_numpy() - 1
        self.texts = csv['Description'].to_list()

    def load_preprocessed_data(self):
        """
        Loads preprocessed and saved data.
        """
        self.preprocessed_texts = read_file(self.cfg.preprocessed_dataset_path +
                                            f'{self.dataset_type}_preprocessed.pickle').preprocessed_text.to_list()#[:1000]

    def preprocess_texts(self):
        """
        Preprocesses texts (applies punctuation removal, converting digits to words, tokenization, stopwords removal).
        """
        if self.cfg.load_preprocessed_data:
            self.load_preprocessed_data()
        else:
            nlp = spacy.load('en_core_web_sm')
            self.preprocessed_texts = []
            start_time, preprocessing_start_time = time.time(), time.time()
            stopwords = nlp.Defaults.stop_words
            for i, text in enumerate(self.texts):
                if i % 1e3 == 0:
                    print(f'Preprocessed {i}/{self.texts_num} texts in {time.time() - start_time} sec')
                    start_time = time.time()
                # punctuation removal
                clean_text = re.sub(r'[^A-Za-z0-9]', ' ', text)
                # converting digits to words
                for k, v in self.cfg.numbers_alias.items():
                    clean_text = re.sub(r'%s' % k, f' {v} ', clean_text)
                    if v in stopwords:
                        stopwords.remove(v)
                # tokenization, stopwords removal
                tokenized_text = []
                for token in nlp(clean_text):
                    token = token.text.strip().lower()
                    if token not in stopwords and len(token) > 1:
                        tokenized_text.append(token)
                self.preprocessed_texts.append(tokenized_text)
            print(f'Preprocessing time: {time.time() - preprocessing_start_time} sec')

            save_file(path=self.cfg.preprocessed_dataset_path + f'{self.dataset_type}_preprocessed.pickle',
                      columns_names=['text', 'preprocessed_text'],
                      columns_content=[self.texts, self.preprocessed_texts])

    def get_hash(self, s, is_subword=False):
        """
        Applies FNV hashing.
        :param s: string to hash
        :return:
        """
        h = np.uint32(2166136261)
        for i in range(len(s)):
            h_ = h ^ np.uint32(np.int8(ord(s[i])))
            h = h_ * 16777619
        if not is_subword:
            h = h % self.cfg.max_vocab_size
        else:
            h = h % self.cfg.bucket_size
            # h += self.cfg.max_vocab_size
        assert h >= 0
        return h

    def add_subwords(self, word):
        subwords_hashes = []
        for step in range(self.cfg.min_n, self.cfg.max_n + 1):
            for i in range(len(word) - step + 1):
                subword = word[i: i + step]

                h = self.get_hash('<' + subword + '>', is_subword=True)
                subwords_hashes.append(h)

                ind_in_words_bucket = self.word2int_bucket[h]

                if ind_in_words_bucket != -1:
                    self.words_bucket[ind_in_words_bucket]['count'] += 1
                else:
                    self.add_to_bucket(h, subword)
        return subwords_hashes

    def add_to_words_(self, h, word, subwords_hashes, bigrams_hashes):
        ind_in_words_ = self.last_index_in_words_
        self.words_[ind_in_words_] = {'word': word, 'count': 1,
                                      'subwords_hashes': subwords_hashes,
                                      'bigrams_hashes': bigrams_hashes}
        self.word2int_[h] = ind_in_words_
        self.last_index_in_words_ += 1

    def threshold(self):
        ids_to_keep = np.asarray([i for i, d in enumerate(self.words_) if d != -1 and d['count'] > self.cfg.min_count])
        self.words_ = [self.words_[i] for i in ids_to_keep]
        self.words_ = self.words_ + [-1] * (self.cfg.max_vocab_size - len(self.words_))

        self.word2int_ = list(np.ones(self.cfg.max_vocab_size) * -1)
        for i, word_info in enumerate(self.words_):
            if word_info != -1:
                h = self.get_hash(word_info['word'])
                self.word2int_[h] = i

    @staticmethod
    def get_fin_arrays(words_, word2int_, words_path, word2int_path):
        words_ = [d for d in words_ if d != -1]
        word2int_ = {i: v for i, v in enumerate(word2int_) if v != -1}
        save_list_as_pickle(words_path, words_)
        save_list_as_pickle(word2int_path, word2int_)

    def add_to_bucket(self, h, subword):
        ind_in_words_ = self.last_index_in_bucket
        self.words_bucket[ind_in_words_] = {'word': subword, 'count': 1}
        self.word2int_bucket[h] = ind_in_words_
        self.last_index_in_bucket += 1

    def aggregate_bigram_hashes(self, h1, h2):
        h1 = np.uint64(h1)
        h = h1 * np.uint64(116049371) + np.uint64(h2)
        h = h % self.cfg.bucket_size
        return np.uint64(h)

    def add_bigrams(self, w_i, text):
        words_pairs = []
        if w_i > 0:
            words_pairs.append((text[w_i - 1], text[w_i]))
        if w_i < len(text) - 1:
            words_pairs.append((text[w_i], text[w_i + 1]))

        bigrams_hashes = []
        for pair in words_pairs:
            h1 = self.get_hash(pair[0]) % self.cfg.bucket_size + self.cfg.max_vocab_size
            h2 = self.get_hash(pair[1]) % self.cfg.bucket_size + self.cfg.max_vocab_size
            h_pair = self.aggregate_bigram_hashes(h1, h2)
            bigrams_hashes.append(h_pair)

            ind_in_words_bucket = self.word2int_bucket[h_pair]

            if ind_in_words_bucket != -1:
                self.words_bucket[ind_in_words_bucket]['count'] += 1
            else:
                self.add_to_bucket(h_pair, f'{pair[0]} {pair[1]}')
        return bigrams_hashes

    def get_arrays_(self):
        """
        Builds vocab.
        """
        if self.cfg.load_arrays:
            self.words_ = read_list(self.cfg.words_path)
            self.word2int_ = read_list(self.cfg.word2int_path)

            self.words_bucket = read_list(self.cfg.words_bucket_path)
            self.word2int_bucket = read_list(self.cfg.word2int_bucket_path)

        else:
            if self.dataset_type == 'train':
                t = time.time()

                for i in range(self.texts_num):
                    if i % 1000 == 0:
                        print(f'text: {i}/{self.texts_num}')
                    text = self.preprocessed_texts[i]

                    # bigram_hashes = self.add_bigrams(text)

                    for w_i, word in enumerate(text):
                        h = self.get_hash(word)
                        subwords_hashes = self.add_subwords(word)
                        bigrams_hashes = self.add_bigrams(w_i, text)

                        ind_in_words_ = self.word2int_[h]

                        if ind_in_words_ != -1:
                            w_info = self.words_[ind_in_words_]

                            if w_info['word'] == word:
                                self.words_[ind_in_words_]['count'] += 1

                            else:
                                while ind_in_words_ != -1:
                                    h += 1
                                    h = h % self.cfg.max_vocab_size
                                    ind_in_words_ = self.word2int_[h]
                                self.add_to_words_(h, word, subwords_hashes, bigrams_hashes)

                        else:
                            self.add_to_words_(h, word, subwords_hashes, bigrams_hashes)

                        filled_vocab_size = self.last_index_in_words_ - 1
                        if filled_vocab_size > self.cfg.thr_for_pruning:
                            self.cfg.min_threshold += 1
                            self.threshold()

                    # self.add_bigrams(text)

                self.threshold()
                self.get_fin_arrays(self.words_, self.word2int_, self.cfg.words_path, self.cfg.word2int_path)
                self.get_fin_arrays(self.words_bucket, self.word2int_bucket, self.cfg.words_bucket_path,
                                    self.cfg.word2int_bucket_path)
                print(f'Arrays getting time {time.time() - t} s')

    def get_encoded_text(self, idx):
        """
        Gets encoded text by idx with precomputed weighted values and their locations in bow vector.
        :param idx: text index in dataset
        :return: encoded text
        """
        ids = []
        for word in self.preprocessed_texts[idx]:
            word_hash = self.get_hash(word)
            word_emb_id = self.word2int_.get(word_hash, None)

            if word_emb_id is not None:
                words_ = self.words_[word_emb_id]
                char_n_grams_hashes = words_['subwords_hashes']
                char_n_grams_ids = [self.cfg.fin_vocab_size + self.word2int_bucket[h] for h in char_n_grams_hashes]
                indexes = [word_emb_id] + char_n_grams_ids

                if self.train_cfg.use_bigrams:
                    bigrams_hashes = words_['bigrams_hashes']
                    bigrams_ids = [self.cfg.fin_vocab_size + self.word2int_bucket[h] for h in bigrams_hashes]
                    indexes += bigrams_ids

                ids.extend(indexes)
        return ids

    def __len__(self):
        """
        Gets dataset length.
        :return: dataset length
        """
        return len(self.preprocessed_texts)

    def __getitem__(self, idx):
        """
        Gets dataset item (encoded text and corresponding label)
        :param idx: index for getting data
        :return: encoded_text and it`s label
        """
        encoded_text = self.get_encoded_text(idx)
        label = self.labels[idx]
        return encoded_text, label
