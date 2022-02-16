

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from collections import OrderedDict, Counter
import logging
import json
import spacy
import string



from brat_scoring.constants import ENCODING, ARG_1, ARG_2, ROLE, TYPE, SUBTYPE, EVENT_TYPE, ENTITIES, COUNT, RELATIONS, ENTITIES, EVENTS, SPACY_MODEL, TRIGGER
from brat_scoring.corpus import Corpus
from brat_scoring.document_brat import DocumentBrat
from brat_scoring.brat import get_brat_files
from brat_scoring.proj_setup import make_and_clear



class CorpusBrat(Corpus):

    def __init__(self, document_class=DocumentBrat, spacy_model=SPACY_MODEL):

        self.document_class = document_class
        self.spacy_model = spacy_model

        Corpus.__init__(self)


    def import_dir(self, path, \
                        n = None,
                        ann_map = None):

        tokenizer = spacy.load(self.spacy_model)

        '''
        Import BRAT directory
        '''

        # Find text and annotation files
        text_files, ann_files = get_brat_files(path)
        file_list = list(zip(text_files, ann_files))
        file_list.sort(key=lambda x: x[1])

        logging.info(f"Importing BRAT directory: {path}")

        if n is not None:
            logging.warn("="*72)
            logging.warn("Only process processing first {} files".format(n))
            logging.warn("="*72)
            file_list = file_list[:n]

        logging.info(f"BRAT file count: {len(file_list)}")

        pbar = tqdm(total=len(file_list), desc='BRAT import')

        # Loop on annotated files
        for fn_txt, fn_ann in file_list:

            # Read text file
            with open(fn_txt, 'r', encoding=ENCODING) as f:
                text = f.read()

            # Read annotation file
            with open(fn_ann, 'r', encoding=ENCODING) as f:
                ann = f.read()

            if ann_map is not None:
                for pat, val in ann_map:
                    ann = re.sub(pat, val, ann)

            # Use filename as ID
            id = os.path.splitext(os.path.relpath(fn_txt, path))[0]

            doc = self.document_class( \
                id = id,
                text = text,
                ann = ann,
                tags = None,
                tokenizer = tokenizer
                )

            # Build corpus
            assert doc.id not in self.docs_
            self.docs_[doc.id] = doc

            pbar.update(1)

        pbar.close()


    def entities(self, include=None, exclude=None, as_dict=False, by_sent=False, entity_types=None):
        """
        Get entities by document
        """

        y = OrderedDict()
        for doc in self.docs(include=include, exclude=exclude):
            y[doc.id] = doc.entities(as_dict=False, by_sent=by_sent, entity_types=entity_types)
        if as_dict:
            pass
        else:
            y = list(y.values())
        return y

    def relations(self, include=None, exclude=None, as_dict=False, by_sent=False, entity_types=None):
        """
        Get relations by document
        """

        y = OrderedDict()
        for doc in self.docs(include=include, exclude=exclude):
            y[doc.id] = doc.relations(by_sent=by_sent, entity_types=entity_types)
        if as_dict:
            pass
        else:
            y = list(y.values())
        return y

    def events(self, include=None, exclude=None, as_dict=False, by_sent=False, event_types=None, entity_types=None):
        """
        Get events by document
        """

        y = OrderedDict()
        for doc in self.docs(include=include, exclude=exclude):
            y[doc.id] = doc.events( \
                                by_sent = by_sent,
                                event_types = event_types,
                                entity_types = entity_types)
        if as_dict:
            pass
        else:
            y = list(y.values())
        return y

    def write_brat(self, path, include=None, exclude=None, \
                                event_types=None, argument_types=None):

        make_and_clear(path, recursive=True)
        for doc in self.docs(include=include, exclude=exclude):
            doc.write_brat(path, \
                            event_types = event_types,
                            argument_types = argument_types)
