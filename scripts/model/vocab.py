import logging
import os
import pickle
import numpy as np
import fasttext


class Vocab:
    PAD_token = 0
    SOS_token = 1
    EOS_token = 2
    UNK_token = 3

    def __init__(self, name, insert_default_tokens=True):
        self.name = name
        self.trimmed = False
        self.word_embedding_weights = None
        self.reset_dictionary(insert_default_tokens)

    def reset_dictionary(self, insert_default_tokens=True):
        self.word2index = {}
        self.word2count = {}
        if insert_default_tokens:
            self.index2word = {self.PAD_token: "<PAD>", self.SOS_token: "<SOS>",
                               self.EOS_token: "<EOS>", self.UNK_token: "<UNK>"}
        else:
            self.index2word = {self.UNK_token: "<UNK>"}
        self.n_words = len(self.index2word)  # count default tokens

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_vocab(self, other_vocab):
        for word, _ in other_vocab.word2count.items():
            self.index_word(word)

    # remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        logging.info('    word trimming, kept %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # reinitialize dictionary
        self.reset_dictionary()
        for word in keep_words:
            self.index_word(word)

    def get_word_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.UNK_token

    def load_word_vectors(self, pretrained_path, embedding_dim=300):
        logging.info("  loading word vectors from '{}'...".format(pretrained_path))

        # initialize embeddings to random values for special words
        init_sd = 1 / np.sqrt(embedding_dim)
        weights = np.random.normal(0, scale=init_sd, size=[self.n_words, embedding_dim])
        weights = weights.astype(np.float32)

        # read word vectors
        word_model = fasttext.load_model(pretrained_path)
        for word, id in self.word2index.items():
            vec = word_model.get_word_vector(word)
            weights[id] = vec

        self.word_embedding_weights = weights

    def __get_embedding_weight(self, pretrained_path, embedding_dim=300):
        """ function modified from http://ronny.rest/blog/post_2017_08_04_glove/ """
        logging.info("Loading word embedding '{}'...".format(pretrained_path))
        cache_path = os.path.splitext(pretrained_path)[0] + '_cache.pkl'
        weights = None

        # use cached file if it exists
        if os.path.exists(cache_path):  #
            with open(cache_path, 'rb') as f:
                logging.info('  using cached result from {}'.format(cache_path))
                weights = pickle.load(f)
                if weights.shape != (self.n_words, embedding_dim):
                    logging.warning('  failed to load word embedding weights. reinitializing...')
                    weights = None

        if weights is None:
            # initialize embeddings to random values for special and OOV words
            init_sd = 1 / np.sqrt(embedding_dim)
            weights = np.random.normal(0, scale=init_sd, size=[self.n_words, embedding_dim])
            weights = weights.astype(np.float32)

            with open(pretrained_path, encoding="utf-8", mode="r") as textFile:
                num_embedded_words = 0
                for line_raw in textFile:
                    # extract the word, and embeddings vector
                    line = line_raw.split()
                    try:
                        word, vector = (line[0], np.array(line[1:], dtype=np.float32))
                        # if word == 'love':  # debugging
                        #     print(word, vector)

                        # if it is in our vocab, then update the corresponding weights
                        id = self.word2index.get(word, None)
                        if id is not None:
                            weights[id] = vector
                            num_embedded_words += 1
                    except ValueError:
                        logging.info('  parsing error at {}...'.format(line_raw[:50]))
                        continue
                logging.info('  {} / {} word vectors are found in the embedding'.format(num_embedded_words, len(self.word2index)))

                with open(cache_path, 'wb') as f:
                    pickle.dump(weights, f)

        return weights
