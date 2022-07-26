


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

from brat_scoring.document import Document
from brat_scoring.brat import get_brat_files, get_brat_files_multi_dir
from brat_scoring.proj_setup import make_and_clear



class Corpus:

    def __init__(self, document_class=Document, spacy_model=SPACY_MODEL):

        self.document_class = document_class
        self.spacy_model = spacy_model

        self.docs_ = OrderedDict()

    def __len__(self):
        return len(self.docs_)

    def __getitem__(self, key):
        return self.docs_[key]

    def __setitem__(self, key, item):
        self.docs_[key] = item

    def __delitem__(self, key):
        del self.docs_[key]

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
        if len(file_list) == 0:
            logging.error(f'''Could not find any brat files at "{path}"''')

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



    def import_predictions(self, txt_path, ann_path, \
                        n = None,
                        ann_map = None,
                        strict_import = True):

        tokenizer = spacy.load(self.spacy_model)

        logging.info(f"Importing predictions")

        '''
        Import BRAT directory
        '''
        # Find text and annotation files
        text_files, ann_files = get_brat_files_multi_dir(txt_path, ann_path)

        file_list = list(zip(text_files, ann_files))
        file_list.sort(key=lambda x: x[0])

        logging.info(f"\tImporting txt directory: {txt_path}")
        logging.info(f"\tImporting ann directory: {ann_path}")

        if n is not None:
            logging.warn("="*72)
            logging.warn("\tOnly process processing first {} files".format(n))
            logging.warn("="*72)
            file_list = file_list[:n]

        # logging.info(f"BRAT file count: {len(file_list)}")
        if len(file_list) == 0:
            logging.error(f'''Could not find any brat files at "{txt_path}"''')

        pbar = tqdm(total=len(file_list), desc='BRAT import')

        # Loop on annotated files
        missing = []
        failure = []
        success = []
        for fn_txt, fn_ann in file_list:

            # Read text file
            with open(fn_txt, 'r', encoding=ENCODING) as f:
                text = f.read()

            # Read annotation file
            if fn_ann is None:
                msg = f"\tMissing annotation file associated with: {fn_txt}"
                logging.warning(msg)
                ann = ""

            else:
                with open(fn_ann, 'r', encoding=ENCODING) as f:
                    ann = f.read()

            if ann_map is not None:
                for pat, val in ann_map:
                    ann = re.sub(pat, val, ann)

            # Use filename as ID
            id = os.path.splitext(os.path.relpath(fn_txt, txt_path))[0]

            doc = self.document_class( \
                id = id,
                text = text,
                ann = ann,
                tags = None,
                tokenizer = tokenizer,
                strict_import = strict_import
                )

            # Build corpus
            assert doc.id not in self.docs_
            self.docs_[doc.id] = doc

            if fn_ann is None:
                missing.append(doc.id)
            elif not doc.import_successful:
                failure.append(doc.id)
            else:
                success.append(doc.id)

            pbar.update(1)

        pbar.close()

        n_missing = len(missing)
        n_failure = len(failure)
        n_success = len(success)

        msg = f"\tCorpus import summary:"
        logging.info(msg)

        msg = f"\t\tMissing count: {n_missing}"
        logging.warning(msg)

        msg = f"\t\tFailure count: {n_failure}"
        logging.warning(msg)

        msg = f"\t\tSuccess count: {n_success}"
        logging.info(msg)


        summary = dict( \
                            n_missing = n_missing,
                            n_failure = n_failure,
                            n_success = n_success,
                            )

        detailed = dict( \
                            missing = missing,
                            failure = failure,
                            success = success,
                            )

        return (summary, detailed)

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
