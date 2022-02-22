
import pandas as pd
import os
from collections import Counter, OrderedDict
import logging
import re
import difflib
import spacy

import brat_scoring.constants as C

from brat_scoring.corpus import Corpus
from brat_scoring.labels import Entity


from brat_scoring.constants_sdoh import LABELED_ARGUMENTS as SDOH_LABELED_ARGUMENTS
SDOH_TRIG = C.OVERLAP
SDOH_SPAN = C.EXACT
SDOH_LABELED = C.LABEL

SCORE_TRIG = C.EXACT
SCORE_SPAN = C.EXACT
SCORE_LABELED = C.LABEL

SPACY_MODEL = C.SPACY_MODEL


def filter_df(df, event_types=None, argument_types=None):

    if event_types is not None:
        df = df[df[C.EVENT].isin(event_types)]

    if argument_types is not None:
        df = df[df[C.ARGUMENT].isin(argument_types)]

    return df

def check_entity_attributes(entity, \
    required_attributes=["type_", "subtype", "char_start", "char_end", "text"]):

    attributes = entity.__dict__.keys()

    for attribute in required_attributes:
        assert attribute in attributes, f'''attribute "{attribute}" is not an attributes {attributes}'''

def insert_total_row(df):

    df_total = df.sum()
    df_total[C.EVENT] = C.OVERALL
    df_total[C.ARGUMENT] = C.OVERALL
    df_total[C.SUBTYPE] = C.OVERALL
    df_total = df_total.to_frame().transpose()
    df_total = PRF(df_total)
    df = pd.concat([df_total, df])

    return df

def get_tokens(text, tokenizer):

    # remove duplicate whitespace
    text = ' '.join(text.split())

    # get tokens
    tokens = tokenizer(text)
    tokens = [x.text for x in tokens]

    return tokens

def get_token_count(text, tokenizer):

    # remove duplicate whitespace
    text = ' '.join(text.split())

    # get token count
    n = len(tokenizer(text))

    return n

def longest_match_len(A, B, verbose=False):

    assert isinstance(A, list)
    assert isinstance(B, list)

    if len(A) > 0:
        assert isinstance(A[0], str)

    if len(B) > 0:
        assert isinstance(B[0], str)

    sequence_matcher = difflib.SequenceMatcher(None, A, B)
    match = sequence_matcher.find_longest_match(0, len(A), 0, len(B))

    A_match = A[match.a:match.a+match.size]
    B_match = B[match.b:match.b+match.size]
    assert A_match == B_match
    assert len(A_match) == match.size

    if verbose:
        print()
        print('A', A)
        print('B', B)
        print('A_match', A_match)
        print('B_match', B_match)
        print('size', match.size)
        print()

    return match.size

def PRF(df):

    df["P"] = df[C.TP]/(df[C.NP].astype(float))
    df["R"] = df[C.TP]/(df[C.NT].astype(float))
    df["F1"] = 2*df["P"]*df["R"]/(df["P"] + df["R"])

    return df

def get_event_counts(events, labeled_args, \
                        score_trig = SCORE_TRIG,
                        score_span = SCORE_SPAN,
                        score_labeled = SCORE_LABELED,
                        tokenizer = None):
    """
    Get histogram of entity labels
    """



    assert score_trig in    [C.EXACT, C.OVERLAP, C.MIN_DIST]
    assert score_span in    [C.EXACT, C.OVERLAP, C.PARTIAL]
    assert score_labeled in [C.EXACT, C.OVERLAP, C.LABEL]

    if score_span == C.PARTIAL:
        assert tokenizer is not None

    counter = Counter()
    for event in events:

        # Get trigger
        trigger = event.arguments[0]

        event_type = trigger.type_

        # iterate over arguments
        for i, argument in enumerate(event.arguments):

            arg_type = argument.type_
            arg_subtype = argument.subtype


            # set count to span-level (each argument counted once)
            c = 1

            # is trigger - always count spans
            if i == 0:
                arg_type = C.TRIGGER

            # is labeled argument - always count spans
            elif argument.type_ in labeled_args:
                pass

            # is span-only argument - can count spans or tokens
            else:
                if score_span == C.PARTIAL:
                    c = get_token_count(argument.text, tokenizer)

                else:
                    pass

            # key for counter
            k = get_event_key(event_type, arg_type, arg_subtype)

            # update counter
            counter[k] += c

    return counter


def get_overlap(i1, i2, j1, j2):
    """
    Get overlap between spans
    """

    A = set(range(i1, i2))
    B = set(range(j1, j2))
    overlap = A.intersection(B)
    overlap = sorted(list(overlap))

    return overlap

def get_overlap_count(i1, i2, j1, j2):
    """
    Determine if any overlap
    """
    overlap = get_overlap(i1, i2, j1, j2)
    return len(overlap)

def has_overlap(i1, i2, j1, j2):
    """
    Determine if any overlap
    """
    overlap = get_overlap(i1, i2, j1, j2)
    return len(overlap) > 0





def compare_entities(gold, predict, entity_scoring=C.EXACT, include_subtype=False,
                tokenizer=None):
    """
    Get histogram of matching entities
    Parameters
    ----------
    gold: Entity
    predict: Entity
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    """

    assert entity_scoring in [C.EXACT, C.PARTIAL, C.OVERLAP, C.LABEL]

    check_entity_attributes(gold)
    check_entity_attributes(predict)

    # assess label match
    type_match = gold.type_ == predict.type_
    subtype_match = gold.subtype == predict.subtype
    if include_subtype:
        type_match = type_match and subtype_match

    y = 0



    if type_match:
        g1, g2 = gold.char_start,    gold.char_end
        p1, p2 = predict.char_start, predict.char_end

        indices_overlap = get_overlap_count(g1, g2, p1, p2)

        # exact match
        # count spans
        if (entity_scoring == C.EXACT) and ((g1, g2) == (p1, p2)):
            y = 1

        # partial match
        # count tokens
        elif (entity_scoring == C.PARTIAL) and indices_overlap:
            gold_tokens =    get_tokens(gold.text,    tokenizer)
            predict_tokens = get_tokens(predict.text, tokenizer)
            y = longest_match_len(gold_tokens, predict_tokens, verbose=False)

        # any overlap match
        # count spans
        elif (entity_scoring == C.OVERLAP) and indices_overlap:
            y = 1

        # subtype labels match
        # count spans
        elif (entity_scoring == C.LABEL) and subtype_match:
            y = 1

    return y



def get_entity_matches(gold, predict, labeled_args, \
                                    score_span = SCORE_SPAN,
                                    score_labeled = SCORE_LABELED,
                                    event_type = None,
                                    tokenizer = None):
    """
    Get histogram of matching entities
    Parameters
    ----------
    gold: list of entities, [Entity, Entity,...]
    predict: list of entities, [Entity, Entity,...]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    """

    assert score_span in    [C.EXACT, C.PARTIAL, C.OVERLAP]
    assert score_labeled in [C.EXACT, C.OVERLAP, C.LABEL]

    if score_span == C.PARTIAL:
        assert tokenizer is not None

    assert isinstance(gold, list)
    assert isinstance(predict, list)

    counter = Counter()

    # iterate over gold
    I = set([])
    J = set([])

    for i, g in enumerate(gold):

        check_entity_attributes(g)

        matching_j = []
        total_v = 0

        # iterate over predicted entities
        for j, p in enumerate(predict):

            check_entity_attributes(p)

            # key for counter
            if event_type is None:
                k = (g.type_, g.subtype)
            else:

                k = get_event_key(event_type, g.type_, g.subtype)

            # assume no matches
            c = 0

            m = (i not in I) and (j not in J)

            # argument types match and are labeled argument
            if (g.type_ == p.type_) and (g.type_ in labeled_args):

                # get entity comparison value as int
                c = compare_entities(g, p, \
                                    entity_scoring = score_labeled,
                                    include_subtype = True,
                                    tokenizer = tokenizer)

            elif (g.type_ == p.type_):

                # get entity comparison value as int
                c = compare_entities(g, p, \
                                    entity_scoring = score_span,
                                    include_subtype = False,
                                    tokenizer = tokenizer)

                m = m or (score_span == C.PARTIAL)

            # include matched values
            if (c > 0) and m:

                I.add(i)
                J.add(j)

                counter[k] += c

    return counter

def get_triggers(events):
    return [event.arguments[0] for event in events]



def span_midpoint(start, end):
    return float(start + end - 1)/2

def get_event_key(event_type, arg_type, arg_subtype):

    if arg_subtype is None:
        arg_subtype = C.NA

    k = (event_type, arg_type, arg_subtype)

    return k

def get_equivalent_triggers_spans(gold, predict, score_trig=SCORE_TRIG):

    assert score_trig in [C.EXACT, C.OVERLAP]

    I = set([])
    J = set([])

    # iterate over gold triggers
    equivalent = []
    for i, g in enumerate(gold):

        check_entity_attributes(g)

        # iterate over predicted triggers
        for j, p in enumerate(predict):

            check_entity_attributes(p)

            # compare triggers
            match = compare_entities(g, p, \
                                        entity_scoring = score_trig,
                                        include_subtype = False)

            # if trigger is match and previously not found, then align
            if match and (i not in I) and (j not in J):
                I.add(i)
                J.add(j)
                equivalent.append((i, j))

    return equivalent

def get_equivalent_triggers_dist(gold, predict, score_trig=SCORE_TRIG):

    assert score_trig in [C.MIN_DIST]

    distances = []

    # iterate over gold triggers
    for i, g in enumerate(gold):

        check_entity_attributes(g)

        # get gold midpoint
        g_midpoint = span_midpoint(g.char_start, g.char_end)

        # iterate over predicted triggers
        for j, p in enumerate(predict):

            check_entity_attributes(p)

            # get predicted midpoint
            p_midpoint = span_midpoint(p.char_start, p.char_end)

            # can only align triggers if the same type
            if g.type_ == p.type_:

                # store distance between gold and predicted
                d = abs(g_midpoint - p_midpoint)
                distances.append((d, i, j))

    # sort distances in ascending order
    distances.sort()

    # using for loop in place of while loop to ensure termination
    n = len(distances)

    equivalent = []
    for _ in range(n):

        # pop closest match
        dist_best, i_best, j_best = distances.pop(0)

        # remove pop'ed indices from list
        distances = [(d, i, j) for (d, i, j) in distances \
                            if (i != i_best) and (j != j_best)]

        # Double check that no triggers get matched twice
        for i_eq, j_eq in equivalent:
            assert i_eq != i_best
            assert j_eq != j_best

        # add matched triggers
        equivalent.append((i_best, j_best))

        if len(distances) == 0:
            break

    return equivalent

def get_equivalent_triggers(gold, predict, score_trig=SCORE_TRIG):

    gold_triggers = get_triggers(gold)
    predict_triggers = get_triggers(predict)


    if score_trig in [C.EXACT, C.OVERLAP]:
        equivalent = get_equivalent_triggers_spans( \
                                    gold = gold_triggers,
                                    predict = predict_triggers,
                                    score_trig = score_trig)


    elif score_trig in [C.MIN_DIST]:
        equivalent = get_equivalent_triggers_dist( \
                                    gold = gold_triggers,
                                    predict = predict_triggers,
                                    score_trig = score_trig)
    else:
        raise ValueError(f"invalid score_trig: {score_trig}")

    return equivalent

def get_event_matches(gold, predict, labeled_args, \
                        score_trig = SCORE_TRIG,
                        score_span = SCORE_SPAN,
                        score_labeled = SCORE_LABELED,
                        tokenizer = None):

    assert score_trig in    [C.EXACT, C.OVERLAP, C.MIN_DIST]
    assert score_span in    [C.EXACT, C.OVERLAP, C.PARTIAL]
    assert score_labeled in [C.EXACT, C.OVERLAP, C.LABEL]

    if score_span == C.PARTIAL:
        assert tokenizer is not None

    equivalent_triggers = get_equivalent_triggers(gold, predict, score_trig=score_trig)

    I = set([])
    J = set([])
    counter = Counter()

    for i, j in equivalent_triggers:

        # get gold and predicted events with equivalent triggers
        g = gold[i]
        p = predict[j]

        gold_trigger =   g.arguments[0]
        gold_arguments = g.arguments[1:]

        predict_trigger =   p.arguments[0]
        predict_arguments = p.arguments[1:]


        k = get_event_key(gold_trigger.type_, C.TRIGGER, gold_trigger.subtype)
        counter[k] += 1

        counter += get_entity_matches( \
                            gold = gold_arguments,
                            predict = predict_arguments,
                            labeled_args = labeled_args,
                            score_span = score_span,
                            score_labeled = score_labeled,
                            event_type = gold_trigger.type_,
                            tokenizer = tokenizer
                            )

    return counter

def get_event_df(nt, np, tp):


    count_dict = OrderedDict([(C.NT, nt), (C.NP, np), (C.TP, tp)])

    cols = [C.EVENT, C.ARGUMENT, C.SUBTYPE]

    counts = Counter()
    for name, counter in count_dict.items():

        for (event_type, arg_type, subtype), c in counter.items():
            k = (event_type, arg_type, subtype, name)
            assert k not in counts
            counts[k] = c

    counts = [list(k) + [v] for k, v in counts.items()]

    df = pd.DataFrame(counts, columns= cols + [C.METRIC, C.COUNT])

    print(nt, np, tp)
    print(df)


    df = pd.pivot_table(df, values=C.COUNT, index=cols, columns=C.METRIC)
    df = df.fillna(0).astype(int)
    df = df.reset_index()

    for c in count_dict.keys():
        if c not in df:
            df[c] = 0
    df = df[cols + list(count_dict.keys())]
    df = PRF(df)
    df = df.fillna(0)
    df = df.sort_values(cols)

    print(df)

    return df

def summarize_event_df(df, name=None):

    count_cols = [C.NT, C.NP, C.TP]

    df_sum = df[count_cols].sum().to_frame().transpose()

    df_sum = PRF(df_sum)


    if name is not None:
        if isinstance(name, str):
            df_sum.insert(0, "name", name)
        elif isinstance(name, (tuple, list)):
            for i, n in enumerate(name):
                df_sum.insert(i, f"col{i}", n)
    return df_sum

def summarize_event_csv(f, name=None):

    df = pd.read_csv(f)
    df = summarize_event_df(df, name=name)

    return df

def summarize_event_dfs(df_dict):

    dfs = []
    for k, df in df_dict.items():
        dfs.append(summarize_event_df(df, name=k))
    df = pd.concat(dfs)

    return df

def summarize_event_csvs(file_dict):

    df_dict = OrderedDict([(name, pd.read_csv(f)) for name, f in file_dict.items()])
    df = summarize_event_dfs(df_dict)

    return df





def score_events(ids, gold, predict, labeled_args, \
                        score_trig = SCORE_TRIG,
                        score_span = SCORE_SPAN,
                        score_labeled = SCORE_LABELED,
                        tokenizer = None):
    '''
    Evaluate predicted events against true events
    Parameters
    ----------
    gold: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    predict: nested list of entities, [[Entity, Entity,...], [Entity, Entity,...]]
    entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
    include_subtype: include subtype in result, as bool
    '''


    assert len(gold) == len(predict)
    assert len(ids) == len(gold)

    nt_corpus = Counter()
    np_corpus = Counter()
    tp_corpus = Counter()

    dfs = []

    # iterate over documents
    for id, g, p in zip(ids, gold, predict):

        nt_doc = get_event_counts(g, \
                                labeled_args = labeled_args,
                                score_trig = score_trig,
                                score_span = score_span,
                                score_labeled = score_labeled,
                                tokenizer = tokenizer)

        np_doc = get_event_counts(p, \
                                labeled_args = labeled_args,
                                score_trig = score_trig,
                                score_span = score_span,
                                score_labeled = score_labeled,
                                tokenizer = tokenizer)

        tp_doc = get_event_matches(g, p, \
                                labeled_args = labeled_args,
                                score_trig = score_trig,
                                score_span = score_span,
                                score_labeled = score_labeled,
                                tokenizer = tokenizer)

        df = get_event_df(nt_doc, np_doc, tp_doc)
        df.insert(0, 'id', id)
        dfs.append(df)

        nt_corpus += nt_doc
        np_corpus += np_doc
        tp_corpus += tp_doc

    df_detailed = pd.concat(dfs)

    df_summary = get_event_df(nt_corpus, np_corpus, tp_corpus)

    return (df_summary, df_detailed)


def get_path(path, description=None, ext='.csv', name='scores'):

    if ext in path:
        f = path
    elif description is None:
        f = os.path.join(path, f'{name}{ext}')
    else:
        f = os.path.join(path, f"{name}_{description}{ext}")

    return f

def score_docs(gold_docs, predict_docs, labeled_args, \
                            score_trig = SCORE_TRIG,
                            score_span = SCORE_SPAN,
                            score_labeled = SCORE_LABELED,
                            output_path = None,
                            description = None,
                            include_detailed = False,
                            spacy_model = SPACY_MODEL,
                            event_types = None,
                            argument_types = None):

    """
    Score entities
    """

    if score_span == C.PARTIAL:
        tokenizer = spacy.load(spacy_model)
    else:
        tokenizer = None


    assert isinstance(gold_docs, dict)
    assert isinstance(predict_docs, dict)

    g = sorted(list(gold_docs.keys()))
    p = sorted(list(predict_docs.keys()))

    assert len(g) == len(p), f"Document count mismatch. Gold doc count={len(g)}. Predict doc count={len(p)}"
    assert g == p, f"Document ids do not match. Gold doc ids={g}. Predict doc ids={p}"


    gold_entities = []
    predict_entities = []
    gold_events = []
    predict_events = []
    ids = []
    for id in gold_docs:
        gold_doc = gold_docs[id]
        predict_doc = predict_docs[id]

        # gold_entities.append(gold_doc.entities())
        # predict_entities.append(predict_doc.entities())

        gold_events.append(gold_doc.events())
        predict_events.append(predict_doc.events())

        ids.append(id)


    """
    Score events
    """
    df_summary, df_detailed = score_events(ids, gold_events, predict_events, \
                            labeled_args = labeled_args,
                            score_trig = score_trig,
                            score_span = score_span,
                            score_labeled = score_labeled,
                            tokenizer = tokenizer)


    if (event_types is not None) or (argument_types is not None):
        df_summary = filter_df(df_summary, \
                                event_types = event_types,
                                argument_types = argument_types)

        df_detailed = filter_df(df_detailed, \
                                event_types = event_types,
                                argument_types = argument_types)


    if score_span != C.PARTIAL:
        df_summary = insert_total_row(df_summary)

    if output_path is not None:

        f = get_path(output_path, description=description, ext='.csv', name='scores')
        df_summary.to_csv(f, index=False)
        logging.info(f'Corpus-level scoring saved to: {f}')

        if include_detailed:
            f = re.sub('\.csv', '_detailed.csv', f)
            df_detailed.to_csv(f, index=False)
            logging.info(f'Document-level scoring saved to: {f}')

    return df_summary



def score_brat(gold_dir, predict_dir, labeled_args, \
                            corpus_class = Corpus,
                            sample_count = None,
                            score_trig = SCORE_TRIG,
                            score_span = SCORE_SPAN,
                            score_labeled = SCORE_LABELED,
                            output_path = None,
                            description = None,
                            include_detailed = False,
                            spacy_model = SPACY_MODEL,
                            event_types = None,
                            argument_types = None):

    logging.info("")
    logging.info(f"Gold importing...")
    gold_corpus = corpus_class()
    gold_corpus.import_dir(gold_dir, n=sample_count)
    logging.info(f"Gold imported")

    logging.info("")
    logging.info(f"Predict importing...")
    predict_corpus = corpus_class()
    predict_corpus.import_dir(predict_dir, n=sample_count)
    logging.info(f"Predict imported")

    assert gold_corpus.doc_count()    > 0, f"Could not find any BRAT files at: {gold_dir}"
    assert predict_corpus.doc_count() > 0, f"Could not find any BRAT files at: {predict_dir}"

    gold_docs =    gold_corpus.docs(as_dict=True)
    predict_docs = predict_corpus.docs(as_dict=True)

    logging.info("")
    logging.info(f"Scoring underway...")
    df = score_docs(gold_docs, predict_docs, \
                            labeled_args = labeled_args,
                            score_trig = score_trig,
                            score_span = score_span,
                            score_labeled = score_labeled,
                            output_path = output_path,
                            description = description,
                            include_detailed = include_detailed,
                            spacy_model = spacy_model,
                            event_types = event_types,
                            argument_types = argument_types)

    logging.info(f"Scoring complete")

    return df


def score_brat_sdoh(gold_dir, predict_dir, output_path, \
                            labeled_args = SDOH_LABELED_ARGUMENTS,
                            corpus_class = Corpus,
                            sample_count = None,
                            score_trig = SDOH_TRIG,
                            score_span = SDOH_SPAN,
                            score_labeled = SDOH_LABELED,
                            description = None,
                            include_detailed = False,
                            loglevel = 'info'):

    if loglevel is not None:
        logging.basicConfig(level=loglevel.upper())

    df = score_brat( \
                        gold_dir = gold_dir,
                        predict_dir = predict_dir,
                        labeled_args = labeled_args, \
                        corpus_class = corpus_class,
                        sample_count = sample_count,
                        score_trig = score_trig,
                        score_span = score_span,
                        score_labeled = score_labeled,
                        output_path = output_path,
                        description = description,
                        include_detailed = include_detailed)

    return df


#
# def get_token_count(tokens, length_max=None):
#
#     token_count = len(tokens)
#
#     if length_max is not None:
#         token_count = min(token_count, length_max)
#     return token_count




# def get_entity_counts(entities, entity_scoring=C.EXACT, include_subtype=False,
#                                                 return_counts_by_entity=False):
#     """
#     Get histogram of entity labels
#     Parameters
#     ----------
#     entities: list of entities, [Entity, Entity,...]
#     entity_scoring: scoring type as str in ["exact", "overlap", "partial"]
#     include_subtype: include subtype in result, as bool
#     """
#
#     assert isinstance(entities, list)
#
#     counter = Counter()
#     counts_by_entity = []
#     for i, entity in enumerate(entities):
#
#         assert isinstance(entity, Entity)
#
#         # key for counter
#         k = tuple([entity.type_])
#         if include_subtype:
#             k = tuple(list(k) + [entity.subtype])
#
#         v = None
#
#         # count spans
#         if entity_scoring in [C.EXACT, C.OVERLAP, C.LABEL]:
#             v = 1
#
#         # count tokens
#         elif entity_scoring in [C.PARTIAL]:
#             v = entity.token_end - entity.token_start
#
#         else:
#             raise ValueError(f"invalid entities scoring: {entity_scoring}")
#
#         assert v is not None
#
#         counter[k] += v
#         counts_by_entity.append(v)
#
#
#     if entity_scoring in [C.EXACT, C.OVERLAP, C.LABEL]:
#         assert sum(counter.values()) == len(entities)
#         assert sum(counts_by_entity) == len(entities)
#
#     if return_counts_by_entity:
#         return counts_by_entity
#     else:
#         return counter



# def get_entity_diff(gold, predict, entity_scoring=C.EXACT, include_subtype=False):
#     """
#     Get histogram of matching entities
#     Parameters
#     ----------
#     gold: list of entities, [Entity, Entity,...]
#     predict: list of entities, [Entity, Entity,...]
#     entity_scoring: scoring type as str in ["exact", "overlap"]
#     include_subtype: include subtype in result, as bool
#     """
#
#     assert entity_scoring in [C.EXACT, C.OVERLAP]
#
#     assert isinstance(gold, list)
#     assert isinstance(predict, list)
#
#     # iterate over gold
#     I_matches = set([])
#     J_matches = set([])
#
#     # iterate over gold entities
#     for i, g in enumerate(gold):
#         assert isinstance(g, Entity) in "text"]
#
#         # iterate over predicted entities
#         for j, p in enumerate(predict):
#             assert isinstance(p, Entity)
#
#             # get entity comparison value as int
#             v = compare_entities(g, p, \
#                                 entity_scoring = entity_scoring,
#                                 include_subtype = include_subtype)
#
#             match = (v > 0) and \
#                     (i not in I_matches) and \
#                     (j not in J_matches)
#
#             # include matched values
#             if match:
#
#                 assert i not in I_matches
#                 assert j not in J_matches
#
#                 I_matches.add(i)
#                 J_matches.add(j)
#
#     # separate matches an difference
#     gold_match,    gold_diff    = separate_matches(gold,    I_matches)
#     predict_match, predict_diff = separate_matches(predict, J_matches)
#
#     return (gold_match, gold_diff, predict_match, predict_diff)

# def get_event_diff(gold, predict, scoring=C.EXACT, include_subtype=False):
#
#     assert scoring in [C.EXACT, C.OVERLAP]
#
#     I_matches = set([])
#     J_matches = set([])
#
#     gold_match = []
#     gold_diff = []
#     predict_match = []
#     predict_diff = []
#
#     for i, g in enumerate(gold):
#
#         gold_trigger =   g.arguments[0]
#         gold_arguments = g.arguments[1:]
#
#         for j, p in enumerate(predict):
#
#             predict_trigger =   p.arguments[0]
#             predict_arguments = p.arguments[1:]
#
#             trigger_match = compare_entities(gold_trigger, predict_trigger, \
#                                         entity_scoring = scoring,
#                                         include_subtype = include_subtype)
#
#             match = trigger_match and \
#                     (i not in I_matches) and \
#                     (j not in J_matches)
#
#             # include matched values
#             if match:
#
#                 assert i not in I_matches
#                 assert j not in J_matches
#
#                 I_matches.add(i)
#                 J_matches.add(j)
#
#                 g_match, g_diff, p_match, p_diff = get_entity_diff( \
#                         gold = gold_arguments,
#                         predict = predict_arguments,
#                         entity_scoring = scoring,
#                         include_subtype = include_subtype)
#
#                 gold_match.extend(      [(gold_trigger,     x) for x in g_match])
#                 gold_diff.extend(       [(gold_trigger,     x) for x in g_diff])
#                 predict_match.extend(   [(predict_trigger,  x) for x in p_match])
#                 predict_diff.extend(    [(predict_trigger,  x) for x in p_diff])
#
#
#     return (gold_match, gold_diff, predict_match, predict_diff)



# def label_name(entity):
#
#     if (entity.subtype is None) or (entity.subtype == entity.type_):
#         return entity.type_
#     else:
#         return f"{entity.type_} ({entity.subtype})"

# def separate_matches(X, match_indices):
#
#     #assert max(match_indices) < len(X)
#
#     # iterate over gold entities
#     match = []
#     diff = []
#     for i, x in enumerate(X):
#         if i in match_indices:
#             match.append(x)
#         else:
#             diff.append(x)
#
#     assert len(match) + len(diff) == len(X)
#
#     return (match, diff)
