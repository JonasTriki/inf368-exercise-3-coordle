'''
lalalalal

This module contains the backend for Coordle.

Classes:
    CordDoc: a class to represent a document, contains uid, title and a 
    dictionary that maps words to integers. The integers denote the count of 
    the word in the text. 

    RecursiveDescentParser: A class that is to be composed in the index. This
    class contains everything to parse a query and obtain results. 

    Index: The index class that is meant to be used by the user. This contains
    methods to build the index. The inde in this module will be stored in RAM,
    thus achieving very fast query times. If you want the index to be stored in
    disk, see coordle_mongobackend.py

    AI_Index: Subclasses Index, almost same API as Index, requires a function
    that takes in a word token and return a list of similar word tokens. 
'''
import numpy as np
import pandas as pd
# from utils import clean_text
from coordle.utils import clean_text
from typing import Iterable, Union, Callable
import nltk
from collections import deque
from collections.abc import Iterable
from itertools import chain
import pickle
from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm
from os import cpu_count
from string import punctuation as PUNCTUATION
import re

class CordDoc:
    '''
    Class for cord documents.
    '''
    def __init__(self, uid: str, title: str=None):
        '''
        Parameters
        -----------
        W: Word2Vec 2D array

        uid: unique id of document

        int_to_word: Arry of words, where indices corresponds to the word_to_int
                    dictionary

        word_to_int: Dictionary where keys are words, and values are integers

        uid: unique string that represents the document, stands for unique
             identity.

        title: optional, title of document

        store_sents: Whether to store string sentences is memory or not.
                     Will be stored self.stringsents
        '''
        self.uid = uid
        self.title = title
        self.wordcounts = dict()

    def __len__(self):
        '''
        Returns length of text that was trained on 
        '''
        return self.len

    def __repr__(self):
        return self.uid

    def __str__(self):
        s = f'uid: {self.uid}\n'
        s += f'title: {self.title[:12]}'
        return s

    def __hash__(self):
        '''
        CordDoc objects are identified by uid
        '''
        return hash(self.uid)

    def __eq__(self, other):
        '''
        CordDoc objects are identified by uid
        '''
        return self.uid == other.uid
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __contains__(self, word):
        return word in self.wordcounts

    def fit(self, text: str, cleaner: Callable=None, **kwargs) -> tuple:
        '''
        Create sentence vectors for all sentences in given text.
        
        Parameters:
        text: Text as string

        cleaner: Text cleaning function, should take in a string as first
                 argument, and return an iterable of cleaned tokens from string.
                 If None is given,defaults to clean_text from bundled utils
                 module

        kwargs: Keyword arguments for cleaner function

        Returns
        --------
        Tuple of self and the cleaned and tokenized sentences
        (self, unique_tokens)
        '''
        if type(text) == str:
            if cleaner is None:
                cleaner = clean_text
            tokens = cleaner(text)
        elif isinstance(text, Iterable):
            tokens = text
        else:
            raise ValueError(f'Unsupported type for text, got f{type(text)}')

        self.len = len(tokens)

        uniques, counts = np.unique(tokens, return_counts=True)
        self.wordcounts = {word:count for word, count in zip(uniques, counts)}
        return self, uniques


class RecursiveDescentParser:
    '''
    Parser class for Coordle used for parsing queries 
    and also searching using the parsed queries. 

    Made to parse query tokens contained
    in deque objects.

    This class is made to be composed in coordle_backend.Index
    '''
    def __init__(self, index: 'Index', or_operator: str='OR', 
                 and_operator: str='AND', difference_operator: str='NOT', 
                 punctuation: str=None):
        '''
        Parameters:
        ------------
        sets: a dictionary where keys correspond to query tokens, and values
              are sets containing CordDoc objects
        '''
        self.index = index
        self.token_to_set = self.index.docmap
        
        self.or_operator = or_operator
        self.and_operator = and_operator
        self.difference_operator = difference_operator

        self.operators = {
            self.or_operator,
            self.and_operator,
            self.difference_operator
        }

        self.punctuation = punctuation
        if self.punctuation is None:
            self.punctuation = PUNCTUATION.replace('(','').replace(')','')

    def get_logical_querytokens(self, query: str):
        query = re.sub(f'[{self.punctuation}]','',query)
        if len(query) == 0:
            return None

        querytokens: list = re.split('([^a-zA-Z0-9])', query)

        # Gotta do this to capture parenthesis
        querytokens = chain.from_iterable([t.split() for t in querytokens])

        q1 = deque(querytokens)
        q2 = deque()
   
        q2.append(q1.popleft())

        while len(q1) > 0:
            token = q1.popleft()
            
            if q2[-1] == '(' or token == ')':
                q2.append(token)
                continue
            
            # If preceeding token was operator
            if q2[-1] in self.operators:
                q2.append(token)
            # If preceeding token was not operator
            else:
                # If current token is not an operator
                if token not in self.operators:
                    q2.append(self.or_operator)
                q2.append(token)
        return q2

    def assert_query(self, querytokens: deque, errmsgs: list) -> bool:
        '''
        Check if query is properly formatted. Returns True if everything is ok,
        else False. 
        '''
        q = querytokens.copy()
        p_list = []
        p_counter = 0 
        flag = True
        curr = q.popleft()
        
        ##### Initialize by checking the first token #####
        
        # Stray closing parenthesis
        if curr == ')':
            p_counter -= 1
            p_list.append(p_counter)
            flag = False
        
        if curr == '(':
            p_counter += 1
            p_list.append(p_counter)
        
        # If query is starting with an operator
        if curr in self.operators:
            errmsgs.append(f'SyntaxError: First token "{curr}" is an operator')
            flag = False
        
        # If querytokens consisted of only a single token
        if len(q) == 0:
            if p_counter != 0:
                errmsgs.append(f'SyntaxError: Stray parenthesis')
                flag = False
            return flag
        ##################################################
        
        prev = curr
        # Runs if more than one token left
        while len(q) > 0:
            curr: str = q.popleft()
            
            if curr == '(':
                p_counter += 1
                p_list.append(p_counter)
            
            if curr == ')':
                p_counter -= 1
                p_list.append(p_counter)
            
            # If curr is operator
            if curr in self.operators:
                # Two succeeding operators
                if prev in self.operators:
                    errmsgs.append(f'SyntaxError: Two succeeding operators "{prev} {curr}"')
                    flag = False
                    
            prev = curr
        
        # Should only be one token left when interpreter is here
        
        # If ending with an operator
        if prev in self.operators:
            errmsgs.append(f'SyntaxError: Last token "{prev}" is an operator')
            flag = False
        
        ###### Check paranthesis' #####
        
        # If unbalanced number of parenthesis'
        if p_counter > 0:
            errmsgs.append(f'SyntaxError: Found stray opening parenthesis')
            flag = False
            
        # Check if any negative values in p_list, implies stray closing 
        # parenthesis
        if any((x < 0 for x in p_list)): 
            errmsgs.append(f'SyntaxError: Found stray closing parenthesis')
            flag = False
        ###############################
        return flag

    @staticmethod
    def _get_in_parenthesis(q: deque):
        '''
        Expects proper query
        
        input should be like:
        white AND (woman NOT man))
        i.e. it should miss opening parenthesis
        
        returns deque
        white AND (woman NOT man)
        '''
        q_temp = deque()
        i = 1
        while i > 0:
            token = q.popleft()
            if token == '(':
                i += 1
            elif token == ')':
                i -= 1
                if i == 0: return q_temp
                if i < 0: raise ValueError('Bad query')
            q_temp.append(token)

    def parenthesis_handler(self, querytokens: deque):
        '''
        Expects proper query
        
        Turns deque
        retarded OR (white AND (woman NOT man)) 
        
        to
        ['retarded', 'OR', ['white', 'AND', ['woman','NOT','man']]]
        '''
        q1 = querytokens.copy()
        q2 = deque()
        while len(q1) > 0:
            token = q1.popleft()
            if token == '(':
                q_temp: deque = self._get_in_parenthesis(q1)
                q2.append(self.parenthesis_handler(q_temp))
            else:
                q2.append(token)
        return q2

    def clean_tokens(self, q: deque, cleaner: Callable = None) -> deque:
        '''
        Given a logical query (that is queries with set operators and 
        parenthesis'), run a text cleaning function on the tokens that are not
        operators or parenthesis' 
        '''

        if cleaner is None:
            cleaner = clean_text

        q2 = deque()
        for token in q:
            if token in '()' or token in self.operators:
                q2.append(token)
            else:
                q2.append(clean_text(token, return_list=False))
        return q2

    # TODO: Think about edge cases 
    # TODO: Maybe use a explicit stack implementation instead of implicit 
    #       to make operator preceedence more modifiable.

    def _preprocess_query(self, query: str) -> tuple:
        '''
        Preprocesses the query before parsing 

        Returns queryqueue, a deque that contains strings and deques. 
        '''
        querytokens: deque = self.get_logical_querytokens(query)
        
        # Assert that query is well formatted
        errmsgs = []

        # Happens if no query tokens
        if querytokens is None:
            return deque(), errmsgs

        if not self.assert_query(querytokens, errmsgs):
            return None, errmsgs

        # "Preprocess tokens" by cleaning tokens that are terms (words)
        querytokens = self.clean_tokens(querytokens)

        # Handle parenthesis 
        queryqueue = self.parenthesis_handler(querytokens)

        return queryqueue, errmsgs

    def __getitem__(self, query: str):
        '''
        Implements fancy syntax: coordle['query here']
        '''
        return self.search(query)

    def search(self, query: Union[str, deque]) -> tuple:
        '''
        TODO: Docs
        '''
        if type(query) == str:
            queryqueue, errmsgs = self._preprocess_query(query)
        elif type(query) == deque:
            queryqueue, errmsgs = query, []
        else:
            raise ValueError(f'Got unsupported type for query, got {type(deque)}')

        # Happens only if queryqueue is empty
        if errmsgs is None:
            return [], [], []

        # If invalid query
        if queryqueue is None:
            return None, None, errmsgs

        # Parse queryqueue
        self.uidcache = {} 

        results = self.parse(queryqueue)

        results_list = np.array(list(results))
        scores = np.array([self.uidcache[doc.uid] for doc in results_list])

        sort_idx = np.argsort(scores)[::-1]
        results_list = results_list[sort_idx]
        scores = scores[sort_idx]

        scores = scores/scores.sum()*100

        return results_list, scores, errmsgs
    
    def parse(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        if len(q) == 0:
            return set()

        results = self._parse_difference(q)
        return results

    def _parse_difference(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        results = self._parse_and(q)
        
        if len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == self.difference_operator:
            q.popleft()
            return results - self._parse_difference(q)
        else:
            return results

    def _parse_and(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        results = self._parse_or(q)
        
        if len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == self.and_operator:
            q.popleft()
            return results & self._parse_and(q)
        else:
            return results

    def _parse_or(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        results = self._parse_term(q)
    
        if len(q) == 0:
            return results
        
        curr_token = q[0]
        if curr_token == self.or_operator:
            q.popleft()
            return results | self._parse_or(q)
        else:
            return results

    def _parse_term(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        curr_token = q.popleft()

        # Implies parenthesis in query
        if type(curr_token) == deque:
            return self.parse(curr_token)

        if curr_token in self.token_to_set:
            results: set = self.token_to_set[curr_token]
            self._tf_idf(results, curr_token)
            return results
        else:
            return set()

    def _tf_idf(self, docs: set, token: str):
        '''
        Given a set CordDoc objects and a token, calculate TF-IDF relevance for
        the objects with respect to the token and then store the value inside 
        the objects.

        The values should be reset to zero after a search
        '''    
        for doc in docs:
            if doc.uid not in self.uidcache:
                self.uidcache[doc.uid] = 0.0

            idf = np.log(len(self.index) / len(docs))
            tf = doc.wordcounts[token] / len(doc)
            self.uidcache[doc.uid] += tf*idf


class Index:
    '''
    Index object for Cord data
    '''
    def __init__(self):
        self.docmap = dict()
        self.rdp = RecursiveDescentParser(self)
        self.len = 0

    def __len__(self):
        '''
        Implements polymorphism for len function
        '''
        return self.len
    
    def __getitem__(self, query: str) -> tuple:
        '''
        Implements fancy syntax: coordle['query here']
        '''
        return self.search(query)

    def add(self, uid: str, title: str, text: Union[str, Iterable]):
        '''
        Adds document to index

        Parameters:
        -------------
        uid: unique identification string

        title: title of document

        text: text of document
        '''
        # Make wordfreqs
        doc = CordDoc(uid=uid, title=title)
        doc, unique_tokens = doc.fit(text)

        self.len += 1

        # Add document to hashmap where keys are unique tokens, and values
        # are sets
        for token in unique_tokens:
            if token not in self.docmap:
                self.docmap[token]=set()
            self.docmap[token].add(doc)

    def build_from_df(self, df: pd.DataFrame, uid: str, title: str, 
                      text: str, use_multiprocessing: bool=False, 
                      workers: int=1, verbose: bool=True, 
                      cleaner: Callable=None):
        '''
        Build index given df, a pd.DataFrame. 

        Parameters
        -----------
        df: pd.DataFrame at least containing separate columns for uid (unique 
            id), title and text.
        
        uid: name of uid column

        title: name of title column

        text: name of text column

        use_multiprocessing: Optional, if True (False by default) it will 
                             preprocess text using threads. 

        workers: Optional, specifies number of workers to use if 
                 use_multiprocessing is True.
        '''
        if cleaner is None:
            cleaner = clean_text

        tqdm_args = {'total':len(df), 'position':0, 'disable':not verbose}

        if use_multiprocessing:
            # Clean texts on multiple cores
            if workers == -1:
                workers = cpu_count()

            if verbose:
                print(f'Text cleaning initilized on {workers} workers')
            with Pool(workers) as pool:
                clean_iterator = tqdm(
                    pool.imap(cleaner, df[text]),
                    desc='Cleaning texts',
                    **tqdm_args
                )
                texts=list(clean_iterator)
        else:
            texts = df[text]
        
        uids = df[uid]
        titles = df[title]

        for uid_, title_, text_ in tqdm(zip(uids, titles, texts), 
                                        desc='Adding to index', **tqdm_args):
            self.add(uid_, title_, text_)

    def extend_with_df(self, df: pd.DataFrame, uid: str, title: str, 
                       text: str, use_multiprocessing: bool=False, 
                       workers: int=1, verbose: bool=True, 
                       cleaner: Callable=None):
        '''Heh'''

    def get_doc(self, uid: str):
        '''
        Get document given uid
        '''
        return self.uid_docmap[uid]

    def search(self, query: Union[str, list], verbose=False) -> tuple:
        '''
        Returns a list of query results given
        query as string or list of strings, also returns tf-idf scores
        '''
        return self.rdp.search(query)


class QueryAppenderIndex(Index):
    '''
    Essentially, uses TF-IDF, but adds similar query tokens 
    to given query using AI, Big Data and Machine Learning $$$
    '''
    def __init__(self, most_similar: Callable, n_similars: int=3):
        super().__init__()
        self.most_similar = most_similar
        self.n_similars = n_similars
    
    def _similar_adder(self, q: deque, token: str):
        '''
        Appends similar words to given token to queue  
        '''
        try:
            similars = [word for word, _ in \
                        self.most_similar(token)[:self.n_similars]]

            for word in similars:
                q.append(self.rdp.or_operator)
                q.append(word)
        except KeyError:
            pass

    def _add_most_similar_tokens(self, q: deque):
        '''
        Assumes proper query 

        Given a list of tokens, for each token in tokens, append the 
        most similar 
        '''

        q1 = q.copy()
        q2 = deque()

        # Initialize by checking first element
        token = q1.popleft()
        q2.append(token)

        if not (token in '()' or token in self.rdp.operators):
            self._similar_adder(q2, token)

        for token in q1:
            q2.append(token)
            if token in '()' or token in self.rdp.operators:
                continue

            # Dont add extra tokens if NOT operator
            if q2[-2] == self.rdp.difference_operator:
                continue

            self._similar_adder(q2, token)

        return q2

    def _preprocess_query_with_ai(self, query: str) -> tuple:
        '''
        Preprocesses the query before parsing 
        '''
        querytokens: deque = self.rdp.get_logical_querytokens(query)

        # Assert that query is well formatted
        errmsgs = []

        # Happens if no tokens
        if querytokens is None:
            return deque(), errmsgs

        if not self.rdp.assert_query(querytokens, errmsgs):
            return None, errmsgs

        # "Preprocess tokens" by cleaning tokens that are terms (words)
        querytokens = self.rdp.clean_tokens(querytokens)

        # This is the "AI" part. This will return querytokens that has gotten
        # appended similar tokens to the already existing tokens 
        querytokens = self._add_most_similar_tokens(querytokens)

        # Handle parenthesis 
        queryqueue = self.rdp.parenthesis_handler(querytokens)

        return queryqueue, errmsgs
 
    def search(self, query: str) -> tuple:
        '''
        Returns a list of query results given
        query as string or list of strings, also returns tf-idf scores
        '''
        queryqueue, errmsgs = self._preprocess_query_with_ai(query)

        if queryqueue is None:
            # Should always be error message if anything is wron with query
            assert len(errmsgs) > 0  
            return None, None, errmsgs

        return self.rdp.search(queryqueue)


        

