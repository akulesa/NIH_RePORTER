import grantminer.config as config
import pandas as pd
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
import re  # For preprocessing
from gensim.models.phrases import Phrases, Phraser

# Load data
def load(year,abstract=False,verbose=False):
    """ Load NIH datasets into pandas dataframes.
    Inputs:
    - year (int): 2018, 2019, 2020
    - abstract (bool):  True, False
    - verbose (bool): True, False
    Outputs:
    - Pandas DataFrame object read from csv file
    """
    if verbose:
        print('Reading filename: '+ str(config.filename(year,abstract)))

    out = pd.read_csv(
        config.filename(year,abstract),
        error_bad_lines=False,
        encoding='latin',
        )

    if verbose:
        print('Number of lines: '+str(out.shape[0]))

    return out

# Load Spacy NLP model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser']) # disabling Named Entity Recognition for speed
print('Spacy en_core_web_sm loaded!')

def lemmatize(doc):
    """ Lemmatizes and removes stopwords.
    Inputs:
        - doc (spacy Doc) [create from spacy pipe function]
    Outputs:
        - txt (str), a string of the input with words lemmatized and stop words removed.
    """
    txt = [token.lemma_ for token in doc if not token.is_stop]
    # Word2Vec uses context words to learn the vector representation of a target word,
    # if a sentence is only one or two words long,
    # the benefit for the training is very small
    if len(txt) > 2:
        return ' '.join(txt)

def clean_text_from_df(df,column,clean_label='clean',verbose=False):
    """ Clean up grant abstracts or other text.
    Inputs:
    - df (Pandas DataFrame), containing the text to be cleaned
    - column (str), the column label containing the text to be cleaned
    - verbose(bool)
    Outputs:
    - df_clean
    """
    df = df.copy()

    # Lowercase and remove non alphanumeric characters
    lowercase = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df[column])

    if verbose:
        print('Lowercased and removed non-alphanumeric characters.')
    # cleaned and lemmatized by spacy NLP model
    df[clean_label] = [lemmatize(doc) for doc in nlp.pipe(lowercase, batch_size=5000)]

    if verbose:
        print('Cleaned and lemmatized.')

    return df

def find_bigrams(df,column,verbose=False):
    """ Find bigrams using gensim Phrases model.
    Inputs:
        - df (Pandas DataFrame), each row should contain a sentence (str)
        - column (str), the column label containing sentence strings
        - verbose (bool)
    Outputs:
        - Trained Gensim Phrases model
    """

    # Create a list of all the sentences in the dataset, and then split them into lists. result is list of lists.
    sentences = [row.split() for row in df[column]]

    # This Phrases function from gensim will count bigrams with min_count=30
    bigram = Phrases(sentences, min_count=30, progress_per=10000)

    return bigram

def freeze_bigram(bg):
    """ Freeze the model. Can no longer update but much smaller and faster.
    Inputs:
        - bg, a trained gensim Phrases model
    Outputs:
        - bigram_frozen, a frozen gensim model using Phraser
    """

    # Phraser will freeze the phrases model and make it much more efficient because we can discard a bunch of stuff from memory
    bigram_frozen = Phraser(bg)

    return bigram_frozen
