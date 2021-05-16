import multiprocessing
import numpy as np
from gensim import matutils
from gensim.models import Word2Vec, KeyedVectors

import grantminer.config as config


def initialize_model(**kw_args):
    """Initialize word2vec model from gensim. Auto-loads defaults from config file.
    Inputs:
        - Keyword arguments for Word2Vec function from Gensim
    Outputs:
        - Word2Vec model
    """
    p = config.w2v_params
    p.update(kw_args)

    w2v_model = Word2Vec(**p)
    return w2v_model

def load(model='full',**kw_args):
    """Load word2vec model or word vectors.
    Inputs:
        - model (str): 'full' or 'vectors', if only load word vectors
    Outputs:
        - word2vec model or word vectors
    """
    if model == 'full':
        return Word2Vec.load(str(config.filepath['w2v_model']))
    elif model == 'vectors':
        return KeyedVectors.load(str(config.filepath['word_vectors']))
    else:
        print('Model must be full or vectors.')

def grant2vec(sentence, weights, word_vectors, print_error=False):
    """ Construct grant vector by a weighted average of the word vectors for all the words in project terms. Returns vector of normalized length.
    Inputs:
        - sentence, list of str (from str.split())
        - weights, dict of weights for each word in sentence
        - word_vectors, KeyedVector trained from word2vec
        - print_error, bool, Optional
    Outputs:
        - (300,) vector with unit length
    """
    arrays = []

    for word in sentence:
        try:
            arrays.append(weights[word]*word_vectors.get_vector(word,norm=True))
        except:
            if print_error:
                print(word+' not found.')
    arrays = np.vstack(arrays).mean(axis=0)
    return matutils.unitvec(arrays)
    # return arrays.mean(axis=0)
