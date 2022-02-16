


import json
import os
import joblib
import re
import shutil
import pandas as pd
from collections import OrderedDict, Counter
import logging
from pathlib import Path
import itertools


from brat_scoring.document import Document
from brat_scoring.brat import write_txt, write_ann


class Corpus:
    '''
    Corpus container (collection of documents)
    '''
    def __init__(self):

        self.docs_ = OrderedDict()

    def __len__(self):
        return len(self.docs_)

    def __getitem__(self, key):
        return self.docs_[key]

    def __setitem__(self, key, item):
        self.docs_[key] = item

    def __delitem__(self, key):
        del self.docs_[key]

    def add_doc(self, doc):
        '''
        Add new document to corpus
        '''

        # Prevent duplicate document IDs
        assert doc.id not in self.docs_, \
        "corpus ids:\n{}\ndoc id:\t{}".format(self.docs_.keys(), doc.id)

        # Add document to corpus
        self.docs_[doc.id] = doc

        return True

    def docs(self, as_dict=False):
        '''
        Get documents
        '''

        # Output documents as dict (no change to output needed)
        if as_dict:
            return self.docs_
        else:
            return list(self.docs_.values())

    def doc_count(self):
        '''
        Get document count
        '''
        return len(self.docs())

    def sentence_count(self):

        count = 0
        for doc in self.docs():
            count += doc.sentence_count()
        return count

    def word_count(self):

        count = 0
        for doc in self.docs():
            count += doc.word_count()
        return count
