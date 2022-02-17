

from collections import Counter
from collections import OrderedDict
import logging
import copy
import pandas as pd
import os


from brat_scoring.constants import EVENT, RELATION, TEXTBOUND, ATTRIBUTE, ENTITIES, RELATIONS, EVENTS, ARGUMENTS, TRIGGER
from brat_scoring.brat import get_annotations, write_txt, write_ann, get_next_index, Textbound, Attribute
from brat_scoring.labels import tb2entities, tb2relations, brat2events
from brat_scoring.brat import Attribute, get_max_id




def ids_to_keep(event_dict, tb_dict, event_types=None, argument_types=None):


    event_ids_keep = set([])
    tb_ids_keep = set([])

    # iterate over events
    for event_id, event in event_dict.items():

        # iterate over arguments in current event, including trigger
        for arg_role, tb_id in event.arguments.items():

            # get text bound for current argument
            tb = tb_dict[tb_id]

            # check for event and argument type equivalence
            event_type_match =    (event_types is None) or    (event.type_ in event_types)
            argument_type_match = (argument_types is None) or (tb.type_ in argument_types)

            # collect ids if match
            if event_type_match and argument_type_match:
                event_ids_keep.add(event_id)
                tb_ids_keep.add(tb_id)

    return (event_ids_keep, tb_ids_keep)


def tokenize_document(text, tokenizer):

    doc = tokenizer(text)

    # get sentences
    sentences = list(doc.sents)

    # remove empty sentences
    sentences = [sent for sent in sentences if sent.text.strip()]

    # iterate over sentences
    tokens = []
    offsets = []
    for sent in sentences:

        # get non whitespace tokens
        sent = [t for t in sent if t.text.strip()]

        # get tokens
        tokens.append([t.text for t in sent])

        # get token offsets
        offsets.append([(t.idx, t.idx + len(t.text)) for t in sent])

    # Check
    assert len(tokens) == len(offsets)
    for tok, off in zip(tokens, offsets):
        assert len(tok) == len(off)
        for t, o in zip(tok, off):
            assert t == text[o[0]:o[1]]

    return (tokens, offsets)



class Document:


    def __init__(self, \
        id,
        text,
        ann,
        tags = None,
        tokenizer = None,
        ):


        # Check text validity
        assert text is not None, "text is None"
        assert isinstance(text, str), "text is not str"
        text_wo_ws = ''.join(text.split())
        assert len(text_wo_ws) > 0, '''text has no non-whitespace chars: "{}"'''.format(repr(text))

        self.id = id
        self.text = text

        self.tags = set([]) if tags is None else tags


        if tokenizer is None:
            self.tokens, self.token_offsets = None, None
        else:
            self.tokens, self.token_offsets = tokenize_document(text, tokenizer)

        self.ann = ann

        # Extract events, text bounds, and attributes from annotation string
        self.event_dict, self.relation_dict, self.tb_dict, self.attr_dict = get_annotations(ann)


    def __str__(self):
        return self.text

    def sentence_count(self):
        if self.tokens is None:
            return None
        else:
            return  len(self.tokens)

    def word_count(self):
        if self.tokens is None:
            return None
        else:
            return sum([len(sent) for sent in self.tokens])

    def entities(self, as_dict=False, by_sent=False, entity_types=None):
        '''
        get list of entities for document
        '''


        entities = tb2entities(self.tb_dict, self.attr_dict, \
                                            as_dict = as_dict,
                                            tokens = self.tokens,
                                            token_offsets = self.token_offsets,
                                            by_sent = by_sent)

        if entity_types is not None:
            entities = [entity for entity in entities if entity.type_ in entity_types]

        return entities

    def relations(self, by_sent=False, entity_types=None):
        '''
        get list of relations for document
        '''



        relations = tb2relations(self.relation_dict, self.tb_dict, self.attr_dict, \
                                            tokens = self.tokens,
                                            token_offsets = self.token_offsets,
                                            by_sent = by_sent)

        if entity_types is not None:
            relations = [relation for relation in relations if \
                                (relation.entity_a.type_ in entity_types) and
                                (relation.entity_b.type_ in entity_types)]

        return relations

    def events(self, by_sent=False, event_types=None, entity_types=None):
        '''
        get list of entities for document
        '''


        events = brat2events(self.event_dict, self.tb_dict, self.attr_dict, \
                                    tokens = self.tokens,
                                    token_offsets = self.token_offsets,
                                    by_sent = by_sent)

        if event_types is not None:

            # filter by event types
            events = [event for event in events if \
                        (event_types is None) or (event.type_ in event_types)]

        if entity_types is not None:


            # filter arguments
            for event in events:
                event.arguments = [arg for arg in event.arguments if \
                        (entity_types is None) or (arg.type_ in entity_types)]

        return events

    def brat_str(self, event_types=None, argument_types=None):



        event_ids_keep, tb_ids_keep = ids_to_keep( \
                                event_dict = self.event_dict,
                                tb_dict = self.tb_dict,
                                event_types = event_types,
                                argument_types = argument_types)
        ann = []

        for tb_id, tb in self.tb_dict.items():
            if tb_id in tb_ids_keep:
                ann.append(tb.brat_str())

        for _, x in self.relation_dict.items():
            ann.append(x.brat_str())

        for event_id, event in self.event_dict.items():
            if event_id in event_ids_keep:
                ann.append(event.brat_str(tb_ids_keep=tb_ids_keep))

        for tb_id, attr in self.attr_dict.items():
            if tb_id in tb_ids_keep:
                ann.append(attr.brat_str())

        ann = "\n".join(ann)

        return ann

    def write_brat(self, path, event_types=None, argument_types=None):

        fn_text = write_txt(path, self.id, self.text)

        ann = self.brat_str( \
                    event_types = event_types,
                    argument_types = argument_types)

        fn_ann = write_ann(path, self.id, ann)

        return (fn_text, fn_ann)
