from os.path import join as join_path
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

'''
spaCy (used for language detection)
-----
To install:
!pip install spacy
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
!pip install spacy-langdetect
'''
import scispacy
import spacy
import en_core_sci_lg # Biomedical word embeddings
from spacy_langdetect import LanguageDetector

class CORD19Data():
    '''
    CORD-19 data preprocessing class
    '''
    def __init__(self, data_dir: str):
        '''Initializes a CORD-19 data preprocessing class
        
        Args:
            data_dir: Raw data directory
        '''
        self.data_dir = data_dir
        
        # Initialize NLP model
        self.nlp = en_core_sci_lg.load(disable=["tagger", "ner"])
        self.nlp.max_length = 2000000
        self.nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        self.nlp_words_to_check = 100
    
    def _load_metadata(self) -> pd.DataFrame:
        '''Loads metadata.csv from raw data into Pandas DataFrame'''
        print('Loading metadata...')
        
        cord_metadata_df = pd.read_csv(join_path(self.data_dir, 'metadata.csv'), dtype={
            'pubmed_id': str,
            'Microsoft Academic Paper ID': str, 
            'doi': str
        })
        
        print('Done!')
        return cord_metadata_df
    
    def _clean_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        '''Cleans given metadata Pandas DataFrame by first
        removing duplicates, then dropping articles without metadata.
        
        Args:
            metadata_df: Metadata Pandas DataFrame to clean
        
        Returns:
            metadata_df: Cleaned metadata Pandas DataFrame
        '''
        print('Cleaning metadata...')
        
        # Remove duplicate cord_uid rows from the metadata.
        metadata_df.drop_duplicates(subset='cord_uid', inplace=True)

        # Drop articles without metadata
        metadata_df = metadata_df[metadata_df.full_text_file.notna()]
        
        print('Done!')
        return metadata_df
    
    def _del_substring_by_indices(self, text: str, rem_indices_pairs: list):
        '''Deletes substring by indices
        
        Args:
            text: Text to delete from
            rem_indices_pairs: Pairs (start, end) of indices to remove from
        
        Returns:
            new_text: Text where indices pairs have been removed
        '''
        rem_indices = set()
        for pairs in rem_indices_pairs:
            rem_indices.update(range(*pairs))
        return ''.join([c for i, c in enumerate(text) if i not in rem_indices])

    def _parse_metadata(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        '''Parses metadata Pandas DataFrame
        
        Args:
            metadata_df: Metadata Pandas DataFrame to parse
        
        Returns:
            cord_df: Pandas DataFrame consisting of columns from metadata Pandas DataFrame
                      and content from article json files.
        '''
        print('Parsing metadata...')
        
        cord_dict = {
            'cord_uid': [],
            'paper_id': [],
            'source': [],
            'is_pmc': [],
            'title': [],
            'body_text': [],
            'doi': [],
            'pubmed_id': [],
            'license': [],
            'abstract': [],
            'publish_time': [],
            'authors': [],
            'journal': [],
            'url': []
        }
        for i in tqdm(range(len(metadata_df))):
            row = metadata_df.iloc[i]

            # Skip rows that does not have full text files
            if not row.has_pmc_xml_parse and not row.has_pdf_parse:
                continue
                
            # Prefer PMC over PDF articles (better quality)
            batch = row.full_text_file
            article_path = join_path(self.data_dir, batch, batch)
            if row.has_pmc_xml_parse:
                article_path = join_path(article_path, 'pmc_json', f'{row.pmcid}.xml.json')
            else:
                sha = row.sha.split('; ')[0] # Use first sha if multiple given
                article_path = join_path(article_path, 'pdf_json', f'{sha}.json')
            
            # Parse body text
            body_text = []
            with open(article_path, 'r') as file:
                content = json.load(file)

                # Body text
                for item in content['body_text']:
                    text = item['text']
                    cite_spans = item['cite_spans']

                    # Remove cite spans from text
                    indices_to_remove = list(map(lambda obj: (obj['start'], obj['end']), cite_spans))
                    text = self._del_substring_by_indices(text, indices_to_remove)

                    body_text.append(text)
            body_text = '\n'.join(body_text)

            # We exclude possible false positive papers that
            # are less than 1000 characters from our dataset
            if len(body_text) < 1000:
                continue
            
            # Append to dict
            cord_dict['body_text'].append(body_text)

            # Append columns from metadata
            cord_dict['cord_uid'].append(row.cord_uid)
            if row.has_pmc_xml_parse:
                cord_dict['paper_id'].append(row.pmcid)
            else:
                cord_dict['paper_id'].append(row.sha)
            cord_dict['source'].append(row.source_x)
            cord_dict['is_pmc'].append(row.has_pmc_xml_parse)
            cord_dict['title'].append(row.title)
            cord_dict['abstract'].append(row.abstract)
            cord_dict['doi'].append(row.doi)
            cord_dict['pubmed_id'].append(row.pubmed_id)
            cord_dict['license'].append(row.license)
            cord_dict['publish_time'].append(row.publish_time)
            cord_dict['authors'].append(row.authors)
            cord_dict['journal'].append(row.journal)
            cord_dict['url'].append(row.url)

        print('Done!')
        return pd.DataFrame(cord_dict)

    def _remove_duplicates(self, df: pd.DataFrame):
        '''Removes duplicate rows with the same abstract/body_text
        
        Args:
            df: DataFrame to remove duplicates from
        
        Returns:
            df: DataFrame without duplicates
        '''
        print('Removing duplicates...')
        df.drop_duplicates(['abstract', 'body_text'], inplace=True)

        print('Done!')
        return df
    
    def _extract_language(self, text: str) -> str:
        '''Extracts the language from the given text using spaCy
        
        Args:
            text: Text to extract language from
        
        Returns:
            lang: Predicted language of text
        '''
        # Extract language using spaCy

        text_first_words = ' '.join(text.split(maxsplit=self.nlp_words_to_check)[:self.nlp_words_to_check])
        lang = self.nlp(text_first_words)._.language['language']
        
        return lang
    
    def _perform_lang_detection(self, df: pd.DataFrame):
        '''Performs language detection on the body_text column of the Pandas DataFrame
        
        Args:
            df: Pandas DataFrame to process
            
        Returns:
            df: Pandas DataFrame with additional language column
        '''
        print('Performing language detection...')
        
        # Extract language
        df['language'] = df.body_text.progress_apply(self._extract_language)

        print('Done!')
        return df
    
    def process_data(self):
        '''Processes the CORD-19 data.
        
        Loads and pre-processes CORD-19 data in specified data directory.
        We take inspiration from Daniel Wolffram's "CORD-19: Create Dataframe" Notebook
        and the "Date updates thread" from the challenge.
        - https://www.kaggle.com/danielwolffram/cord-19-create-dataframe
        - https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/137474
        
        Returns:
            cord_df: Pandas DataFrame with processed CORD-19 data
        '''
        # Perform pre-processing
        metadata_df = self._load_metadata()
        metadata_df = self._clean_metadata(metadata_df)
        cord_df = self._parse_metadata(metadata_df)
        cord_df = self._remove_duplicates(cord_df)
        cord_df = self._perform_lang_detection(cord_df)

        return cord_df