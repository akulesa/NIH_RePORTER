# config.py
from pathlib import Path
import multiprocessing

# Paths
data_dir = Path('../data/')
model_dir = Path('../models/')
filepath = dict()
filepath['raw'] = data_dir / 'raw'
filepath['processed']= data_dir / 'processed'
filepath['clean'] = data_dir / 'clean'
filepath['bigram_frozen'] = model_dir / 'bigram_frozen.pkl'
filepath['bigram'] = model_dir / 'bigram.pkl'
filepath['w2v_model'] = model_dir / 'w2v.pkl'
filepath['word_vectors'] = model_dir / 'word_vectors.pkl'

# Filenames
def filename(year,abstract=False,fpath='raw'):
    """ Return filenames of data from NIH Project RePORTER output
    Inputs:
    - year (int): 2018, 2019, 2020
    - abstract (bool):  True, False
    - fpath (str): raw, processed, clean
    Outputs:
    - filename (str)
    """
    prefix = 'RePORTER_PRJ'
    if abstract:
        mid = 'ABS_C_FY'
    else:
        mid = '_C_FY'
    suffix = str(year)+'_new.csv'
    return filepath[fpath] / (prefix+mid+suffix)

# Model configuration
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_params = dict(
    min_count=20,
    window=5,
    vector_size=300,
    sample=6e-5,
    alpha=0.03,
    min_alpha=0.0007,
    negative=20,
    workers=cores-1,
)
