


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


def include_keep(tags, include):

    # assume keep is true by default
    keep = True

    # exclude labels provided
    if (include is not None):

        # require all include tags to be present
        if not include.issubset(tags):
            keep = False

    return keep


def exclude_keep(tags, exclude):

    # assume keep is true by default
    keep = True

    # exclude labels provided
    if (exclude is not None):

        # at least some overlap between exclude and tags
        if len(exclude.intersection(tags)) > 0:
            keep = False

    return keep


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

    def doc_filter(self, include=None, exclude=None):
        '''
        Get filtered set of documents
        '''

        if isinstance(include, str):
            include = [include]

        if isinstance(exclude, str):
            exclude = [exclude]

        if (include is not None):
            include = set(include)

        if (exclude is not None):
            exclude = set(exclude)

        docs_out = OrderedDict()
        for id, doc in self.docs_.items():

            # go to document tags
            tags = doc.tags
            if tags is None:
                tags = set([])
            if not isinstance(tags, set):
                tags = set(tags)

            keep = True
            keep = keep and include_keep(tags, include)
            keep = keep and exclude_keep(tags, exclude)

            if keep:
                docs_out[id] = doc

        if (include is not None) or (exclude is not None):
            logging.info('Document filter')
            logging.info('\tinclude:         {}'.format(include))
            logging.info('\texclude:         {}'.format(exclude))
            logging.info('\tcount, all:      {}'.format(len(self)))
            logging.info('\tcount, filtered: {}'.format(len(docs_out)))

        return docs_out


    def id2stem(self, id):
        '''
        Convert document ID to filename stem
        '''
        return id

    def docs(self, as_dict=False, include=None, exclude=None):
        '''
        Get documents
        '''

        # Get filtered documents
        docs = self.doc_filter(include=include, exclude=exclude)

        # Output documents as dict (no change to output needed)
        if as_dict:
            pass
        else:
            docs = list(docs.values())

        return docs

    def ids(self, as_stem=False, include=None, exclude=None):
        '''
        Get tokenized documents
        '''
        ids = []
        for doc in self.docs(as_dict=False, include=include, exclude=exclude):

            id = doc.id
            if as_stem:
                id = self.id2stem(id)
            ids.append(id)

        return ids

    def doc_count(self, include=None, exclude=None):
        '''
        Get document count
        '''
        return len(self.docs(include=include, exclude=exclude))


    def sentence_count(self, include=None, exclude=None):

        count = 0
        for doc in self.docs(include=include, exclude=exclude):
            count += doc.sentence_count()
        return count

    def word_count(self, include=None, exclude=None):

        count = 0
        for doc in self.docs(include=include, exclude=exclude):
            count += doc.word_count()
        return count
