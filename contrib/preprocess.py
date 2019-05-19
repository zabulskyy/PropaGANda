import tokenize_uk
import numpy as np
import io


class Preprocess:
    def __init__(self, config):
        self.max_words = config['max_words']

        self.ld = get_lemma_dict(config['lemma_dict_path'])
        print('Lemma dict loaded!')

        self.stop_words = get_stop_words(config['stopwords_path'])
        print('Stop words loaded!')

        self.embeddings, self.id2word, self.word2id = load_vec(config['uk_vec_path'], config['nmax'])
        print('Embeddings loaded!')

    def preprocess_sent(self, s):
        s = str(s).lower()
        words = tokenize_uk.tokenize_words(s)
        words = [word for word in words if word not in self.stop_words]
        words = [self.ld[word] if word in self.ld else word for word in words]
        words = [self.emb[self.word2id[word]] for word in words if word in self.word2id]
        words = np.array(words)
        if words.shape[0] > self.max_words:
            words = np.array([])
        return words

    def __call__(self, s):
        return self.preprocess_sent(s)


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


def get_stop_words(path):
    stop_words = set()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line[0] != '*':
                stop_words.add(line.strip())
    return stop_words


def get_lemma_dict(path):
    lemma_dict = dict()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            l = line.split()
            lemma_dict[l[0]] = l[1]
    return lemma_dict