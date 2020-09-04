import logging
import os
import pickle

import lmdb
import pyarrow

from model.vocab import Vocab


def build_vocab(name, dataset_list, cache_path, word_vec_path=None, feat_dim=None):
    logging.info('  building a language model...')
    if not os.path.exists(cache_path):
        lang_model = Vocab(name)
        for dataset in dataset_list:
            logging.info('    indexing words from {}'.format(dataset.lmdb_dir))
            index_words(lang_model, dataset.lmdb_dir)

        if word_vec_path is not None:
            lang_model.load_word_vectors(word_vec_path, feat_dim)

        with open(cache_path, 'wb') as f:
            pickle.dump(lang_model, f)
    else:
        logging.info('    loaded from {}'.format(cache_path))
        with open(cache_path, 'rb') as f:
            lang_model = pickle.load(f)

        if word_vec_path is None:
            lang_model.word_embedding_weights = None
        elif lang_model.word_embedding_weights.shape[0] != lang_model.n_words:
            logging.warning('    failed to load word embedding weights. check this')
            assert False

    return lang_model


def index_words(lang_model, lmdb_dir):
    lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    txn = lmdb_env.begin(write=False)
    cursor = txn.cursor()

    for key, buf in cursor:
        video = pyarrow.deserialize(buf)

        for clip in video['clips']:
            for word_info in clip['words']:
                word = word_info[0]
                lang_model.index_word(word)

    lmdb_env.close()
    logging.info('    indexed %d words' % lang_model.n_words)

    # filtering vocab
    # MIN_COUNT = 3
    # lang_model.trim(MIN_COUNT)

