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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
from tensorflow.keras.utils import to_categorical, Sequence
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from gensim.models.callbacks import CallbackAny2Vec
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

def tokenize_texts(texts: np.ndarray, max_vocab_size: int):
    '''Tokenizes the texts using Tensorflows Tokenizer class

    Args:
        texts: Texts to tokenize
        max_vocab_size: Maximum vocabulary size. Used to limit the number of words when creating data
    
    Returns:
        (data, word_to_idx, idx_to_word): Tuple with tokenized text data, 
        mapping from word to index and mapping to index to word
    '''
    # Tokenize the texts
    print('Tokenizing texts...')
    tokenizer = Tokenizer(max_vocab_size)
    tokenizer.fit_on_texts(tqdm(texts, unit='text'))
    
    # Extract word dictionary
    word_to_idx = tokenizer.word_index
    idx_to_word = tokenizer.index_word
    
    # Create data
    print('Creating data matrix...')
    data = tokenizer.texts_to_sequences(tqdm(texts, unit='text'))
    
    # Old below
    #data = pad_sequences(texts_seq, max_vocab_size)
    #data = np.array([[word_to_idx[word] for word in text_to_word_sequence(text)] for text in tqdm(texts, unit='text')])
    
    return data, word_to_idx, idx_to_word

class TokenizedSkipgramDataGenerator(Sequence):
    '''
    Data generator for genering tokenized skipgram word pairs ((target, context) -> positive (1) or negative(0)).
    Inherits tensorflow.keras.utils.Sequence
    '''
    def __init__(self,
                 tokenized_data: list,
                 vocab_size: int,
                 sampling_window_size: int,
                 num_negative_samples: int,
                 corpus_batch_size: int,
                 pairs_batch_size: int,
                 categorical_pairs: bool = False,
                 shuffle: bool = True):
        '''Initialization of the tokenized skipgram data generator
        
        Args:
            tokenized_data: Tokenized data. Each item in the list should contain a tokenized text corpus
            vocab_size: Vocabularty size
            sampling_window_size: Sampling window size
            num_negative_samples: Number of negative samples to generate
            corpus_batch_size: Number of tokenized text corpuses to process per epoch
            pairs_batch_size: Number of skipgram word pairs to process at once per epoch
            categorical_pairs: Whether to use categorical (one-hot encoded) target/context pairs. Default: False
            shuffle: Whether to shuffle the data at generation. Default: True
        '''
        self.tokenized_data = tokenized_data
        self.vocab_size = vocab_size
        self.sampling_window_size = sampling_window_size
        self.num_negative_samples = num_negative_samples
        self.corpus_batch_size = corpus_batch_size
        self.pairs_batch_size = pairs_batch_size
        self.categorical_pairs = categorical_pairs
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        '''
        Creates skipgram word pairs and updates indices after each epoch
        '''
        corpus_indices = np.random.choice(len(self.tokenized_data), size=self.corpus_batch_size, replace=False)
        self.skipgram_pairs, self.skipgram_labels = [], []
        for i in corpus_indices:
            corpus_pairs, corpus_labels = skipgrams(
                self.tokenized_data[i],
                self.vocab_size,
                window_size=self.sampling_window_size,
                negative_samples=self.num_negative_samples
            )
            
            self.skipgram_pairs += corpus_pairs
            self.skipgram_labels += corpus_labels
        
        # Convert to numpy
        self.skipgram_pairs = np.array(self.skipgram_pairs)
        self.skipgram_labels = np.array(self.skipgram_labels)
        
        self.num_skipgram_pairs = len(self.skipgram_pairs)
        
        # One-hot encode pairs if needed
        if self.categorical_pairs:
            self.skipgram_pairs = to_categorical(self.skipgram_pairs.T, self.vocab_size + 1)
        
        # Create indices
        self.indices = np.arange(self.num_skipgram_pairs)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return self.num_skipgram_pairs // self.pairs_batch_size
    
    def __getitem__(self, batch_nr: int):
        '''Generate one batch (pairs, labels) of data
        
        Args:
            batch_nr: Batch number
        
        Returns:
            (pairs, labels): Batch of skipgram word pairs and labels
        '''
        return self.get_batch(batch_nr)

    def _get_batch_indices(self, batch_nr: int):
        '''Gets indices for a given batch number
        
        Args:
            batch_nr: Batch number
            
        Returns:
            indices: Indices of the current batch
        '''
        return self.indices[batch_nr * self.pairs_batch_size:(batch_nr + 1) * self.pairs_batch_size]
    
    def get_batch(self, batch_nr: int):
        '''Gets a batch of data (pairs, labels) containing batch_size samples 
        
        Args:
            batch_nr: Batch number
            
        Returns:
            (pairs, labels): Batch of skipgram word pairs and labels
        '''
        # Generate indices of the batch
        batch_indices = self._get_batch_indices(batch_nr)
        
        # Get batch
        if self.categorical_pairs:
            X_batch = self.skipgram_pairs[:, batch_indices]
        else:
            X_batch = self.skipgram_pairs[batch_indices].T
        y_batch = self.skipgram_labels[batch_indices]
        
        # [None] value fixes the following error (assuming tf version < 2.2):
        # https://stackoverflow.com/questions/59317919/warningtensorflowsample-weight-modes-were-coerced-from-to
        return list(X_batch), y_batch, [None]

def fix_authors(authors: str, authors_sep: str = ';', name_sep: str = ','):
    '''
    TODO: Docs
    '''
    authors_split = authors.split(authors_sep)
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

