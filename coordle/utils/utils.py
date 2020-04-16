'''
This module contains general use utility functions
'''
import os
from os.path import join as join_path
import json
import re 
from string import punctuation as PUNCTUATION
from nltk.corpus import stopwords as _stopwords
from typing import Union
import pandas as pd
import numpy as np
from functools import reduce
from typing import Union, Iterable, Callable, Any
import pickle
from tqdm.notebook import tqdm
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim.models.callbacks import CallbackAny2Vec

# Needed to make multiprocessing work when cleaning on multiple cores
next(wn.words())

def set_kaggle_env_keys(kaggle_path: str = 'kaggle.json') -> None:
    '''Sets environment keys required by the KaggleApi
    
    Args:
        kaggle_path: Path to the Kaggle json file
    '''
    with open(kaggle_path, 'r') as file:
        kaggle_json = json.load(file)

        # Set OS environment variables
        os.environ['KAGGLE_USERNAME'] = kaggle_json['username']
        os.environ['KAGGLE_KEY'] = kaggle_json['key']

def clean_text(text: str, stopwords: set=None, punctuation: str=None,
               return_list: bool=True) -> Union[str, list]:
    '''
    Cleans text by turning to lower case, removing punctuations and stopwords
    
    Parameters:
    --------------
    text: String to be cleaned 
    
    stopwords: Optional set of stopwords (as strings). If None is given, the 
               stopwords will be acquired from 
               nltk.corpus.stopwords.words('english')
               
    punctuation: Optional string punctuations. If None is given, the 
                 punctuations will be acquired from strings.punctuations

    return_list: Optional, will return list of tokens of True (True by default),
                 else it returns a string
    
    Returns:
    --------
    If return_list is True: list
    else: string
    '''
    if stopwords is None:
        stopwords = set(_stopwords.words('english'))
        
    if punctuation is None:
        punctuation = PUNCTUATION

    # Lower case 
    text = text.lower()
    # Replace newlines with spaces
    text = text.replace('\n',' ')
    # Remove punctuations
    text = re.sub(f'[{punctuation}]','',text)

    # Filter function, remove stopwords, numbers
    # and strings that have length 1 or less
    f = lambda t: (t not in stopwords) and \
                  (not t.isnumeric()) and \
                  (len(t) > 1)

    # Lemmatize first, then filter
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in text.split()]
    tokens = [t for t in tokens if f(t)]
    
    if return_list:
        return tokens
    else:
        return str.join(' ', tokens)

def fix_authors(authors: str, authors_sep: str = ';', name_sep: str = ','):
    '''Fixes long authors string
    
    Args:
        authors: Authors string (separated by authors_sep)
        authors_sep: Delimiter to use between authors
        name_sep: Delimiter to use between names for each author
    
    Returns:
        fixed_authors: Fixed authors string
    '''
    authors_split = str(authors).split(authors_sep)
    if len(authors_split) > 2:

        # Use first authors last name + et al.
        return f'{authors_split[0].split(name_sep)[0]} et al.'
    else:
        
        # Separate authors using comma (,)
        return ', '.join(authors_split)

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, output_dir: str, prefix: str, start_epoch: int = 1):
        self.output_dir = output_dir
        self.prefix = prefix
        self.epoch = start_epoch

    def on_epoch_end(self, model):
        output_path = join_path(self.output_dir, f'{self.prefix}_epoch_{self.epoch}.model')
        model.save(output_path)
        self.epoch += 1