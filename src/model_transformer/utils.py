import re
import glob
import json
import random
import numpy as np
import pandas as pd
import spacy

from collections import defaultdict
from copy import deepcopy


def get_predicate_tokens(annotation, triple):
    """ Finds the predicate token from an annotated triple.

        params:
        dict annotation:    loaded annotation file (see load_annotations)
        list triple:        contains lists of subject, predicate, object, negation and perspective annotations

        returns:            predicate token as string if there is an annotation, None if there is no annotation
    """
    if triple[1]:
        turn = triple[1][0][0]
        start = triple[1][0][1]
        end = triple[1][-1][0]
        return ' '.join(annotation['tokens'][turn][start:end + 1])
    else:
        return None


def find_unique_predicates(path):
    """ Obtains the predicate token from the annotation and adds it to a set of all unique predicates in the data

        params:
        str path:    name of directory containing annotations

        returns:    set of unique predicates
    """
    annotations = load_annotations(path)
    unique_predicates = set()
    for ann in annotations:
        for triple in ann['annotations']:
            pred = get_predicate_tokens(ann, triple)
            if pred is not None:
                unique_predicates.add(pred)
    return unique_predicates


def load_annotations(path, remove_unk=True, keep_skipped=False):
    """ Reads all annotation files from path. By default, it filters skipped
        files and removes the [unk] tokens appended at the end of each turn.

        params:
        str path:           name of directory containing annotations
        bool remove_unk:    whether to remove [unk] tokens (default: True)
        bool keep_skipped:  whether to keep skipped annotations (default: False)

        returns:    list of annotations dicts
    """
    annotations = []
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)

            if data['skipped'] and not keep_skipped:
                continue

            if remove_unk:
                data['tokens'] = [[t for t in turn if t != '[unk]'] for turn in data['tokens']]

            annotations.append(data)

    return annotations

def cleaning(list_predicates):
    """
    Cleans a list of predicates 
    """
    nlp = spacy.load('en_core_web_sm')
    cleaned = set()

    for predicate in list(list_predicates):
        working = nlp(predicate)
        subset = []

        for token in working:
        # don't include punctuation marks 
            if token.pos_ == "PUNCT":
                continue

            subset.append(token.lemma_)

            # stem all the shortened versions of to be and to have 
            if len(subset) >= 2:
                for i, sep in enumerate(subset):
                # change 'm 
                    if (subset[-1]  == "m") & (subset[-2] == "'") :
                        subset.pop(-2)
                        subset[-1] = 'be'
                    # change 've 
                    elif (subset[-1]  == "ve") & (subset[-2] == "'") :
                        subset.pop(-2)
                        subset[-1] = 'have'
                
                # change 's 
            if subset[-1] == "'s" or subset[-1] == "m" or subset[-1] == "re":
                subset[-1] = 'be'
            # take out speaker since these are subj or obj 
            if subset[-1] == "SPEAKER1" or subset[-1] == "SPEAKER2" or subset[-1] == "speaker1":
                subset.pop(-1)
        cleaned.add(' '.join(subset))
    
    return cleaned


def triple_to_bio_tags(annotation, arg, lookup):
    """ Converts the token indices of the annotations to a vector of BIO labels
        for an argument.

        params:
        dict annotation:    loaded annotation file (see load_annotations)
        int arg:            argument to create tag sequence for (subj=0, pred=1, obj=2)
        dict lookup:        dict with unique predicate as key and tuple (B-tag, I-tag) as value

        returns:    ndarray with BIO labels
    """
    # Determine length of dialogue
    turns = annotation['tokens']
    triples = annotation['annotations']
    num_tokens = sum([len(turn) + 1 for turn in turns])  # +1 for <eos>

    # Create vector same size as dialogue
    mask = np.zeros(num_tokens, dtype=np.uint8)

    # Label annotated arguments as BIO tags
    for triple in triples:
        if arg == 1:
            pred = get_predicate_tokens(annotation, triple)
            if pred is not None:
                #This use of predicate should also go through cleaning and lemmatization
                pred = cleaning([pred])
                pred = list(pred)[0]

                b_tag, i_tag = lookup[pred]
                for j, (turn_id, token_id) in enumerate(triple[arg]):
                    k = sum([len(t) + 1 for t in turns[:turn_id]]) + token_id  # k = index of token in dialogue
                    mask[k] = b_tag if j == 0 else i_tag
        else:
            for j, (turn_id, token_id) in enumerate(triple[arg]):
                k = sum([len(t) + 1 for t in turns[:turn_id]]) + token_id  # k = index of token in dialogue
                mask[k] = 1 if j == 0 else 2
    return mask


def bio_tags_to_tokens(tokens, mask, bio_lookup, predicate=False, one_hot=False):
    """ Converts a vector of BIO-tags into spans of tokens. If BIO-tags are one-hot encoded,
        one_hot=True will first perform an argmax to obtain the BIO labels.

        params:
        list tokens:        list of subwords or tokens (as tokenized by Albert/AutoTokenizer)
        ndarray mask:       list of bio labels (one for each subword or token in 'tokens')
        dict bio_lookup:    dict with B-tag as key and abstract predicate as value
        bool predicate:     whether the conversion is done for predicates or not
        bool one_hot:       whether to interpret mask as a one-hot encoded sequence of shape |sequence|x3
    """
    out = []
    span = []

    for i, token in enumerate(tokens):
        pred = mask[i]

        # Reverse one-hot encoding (optional)
        if one_hot:
            pred = np.argmax(pred)

        if pred % 2 == 1:  # B
            if predicate:
                abstr_pred = [bio_lookup[pred]]
                out.append(abstr_pred[0])
            else:
                span = re.sub('[^\w\d\-\']+', ' ', ''.join(span)).strip()
                span = span.replace('SPEAKER', ' SPEAKER').replace('speaker', ' speaker').strip()
                out.append(span)
                span = [token]

        elif pred != 0 and pred % 2 == 0:  # I
            if predicate:
                continue
            else:
                span.append(token)
    
    if span:
        span = re.sub('[^\w\d\-\']+', ' ', ''.join(span)).strip()
        span = span.replace('SPEAKER', ' SPEAKER').replace('speaker', ' speaker').strip()
        out.append(span)

    # Remove empty strings and duplicates
    return set([span for span in out if span.strip()])


def extract_triples(annotation, neg_oversampling=7, contr_oversampling=0.7, ellipsis_oversampling=3):
    """ Extracts plain-text triples from an annotation file and samples 'negative' examples by
        crossover. By default, the function will over-extract triples with negative polarity and
        elliptical constructions to counter class imbalance.

        params:
        dict annotation:            loaded annotation file (see load_annotations)
        int neg_oversampling:       how much to over-sample triples with negative polarity
        float contr_oversampling:   how much to sample contrast/invalid triples relative to true triples
        int ellipsis_oversampling:  how much to over-sample elliptical triples
    """
    turns = annotation['tokens']
    triple_ids = [t[:4] for t in annotation['annotations']]

    arguments = defaultdict(list)
    triples = []
    labels = []

    # Oversampling of elliptical triples
    for triple in deepcopy(triple_ids):
        subj_obj_turns = set([i for i, _ in triple[0] + triple[2]])
        if len(subj_obj_turns) > 1:
            triple_ids += [triple] * int(ellipsis_oversampling)

    # Extract 'True' triples
    for subj, pred, obj, polar in triple_ids:

        subj = ' '.join(turns[i][j] for i, j in subj) if subj else ''
        pred = ' '.join(turns[i][j] for i, j in pred) if pred else ''
        obj = ' '.join(turns[i][j] for i, j in obj) if obj else ''

        if subj or pred or obj:

            if not polar:
                triples += [(subj, pred, obj)]
                labels += [1]
            else:
                triples += [(subj, pred, obj)] * neg_oversampling  # Oversampling negative polarities
                labels += [2] * neg_oversampling

            arguments['subjs'].append(subj)
            arguments['preds'].append(pred)
            arguments['objs'].append(obj)

    # Skip if the annotation file was blank
    if not triples:
        return [], [], []

    # Sample fake contrast examples (invalid extractions)
    n = int(len(triples) * contr_oversampling)
    for i in range(50):
        s = random.choice(arguments['subjs'])
        p = random.choice(arguments['preds'])
        o = random.choice(arguments['objs'])

        # Ensure samples are new (and not actually valid!)
        if (s, p, o) not in triples and s and p and o:
            triples += [(s, p, o)]
            labels += [0]
            n -= 1

        # Create as many fake examples as there were 'real' triples
        if n == 0:
            break

    return turns, triples, labels


def pronoun_to_speaker_id(token, turn_idx):
    # Even turns -> speaker1
    if turn_idx % 2 == 0:
        if token in ['i', 'me', 'myself', 'we', 'us', 'ourselves']:
            return 'SPEAKER1'
        elif token in ['my', 'mine', 'our', 'ours']:
            return "SPEAKER1's"
        elif token in ['you', 'yourself', 'yourselves']:
            return 'SPEAKER2'
        elif token in ['your', 'yours']:
            return "SPEAKER2's"
    else:
        if token in ['i', 'me', 'myself', 'we', 'us', 'ourselves']:
            return "SPEAKER2"
        elif token in ['my', 'mine', 'our', 'ours']:
            return "SPEAKER2's"
        elif token in ['you', 'yourself', 'yourselves']:
            return 'SPEAKER1'
        elif token in ['your', 'yours']:
            return "SPEAKER1's"
    return token


def speaker_id_to_speaker(string, speaker1, speaker2):
    return string.replace('SPEAKER1', ' ' + speaker1).replace('SPEAKER2', ' ' + speaker2).strip()


if __name__ == "__main__":
    anns = load_annotations('[LOCATION OF TRAINVAL FOLDER]/trainval')
    unique = find_unique_predicates('[LOCATION OF TRAINVAL FOLDER]/trainval')
    unique_list = list(unique)
    first_half = unique_list[:int(len(unique_list)/2)]
    second_half = unique_list[int(len(unique_list)/2):]
    print(first_half)
    print(second_half)
    bio_dict = {(3,4): first_half, (5,6): second_half}
    bio_lookup = {3: 'like', 5: 'do'}
    lookup = {}
    for key, value in bio_dict.items():
        for pred in value:
            lookup[pred] = key
    print(lookup)
    tokens, labels = [], []
    for ann in anns:
        # Map triple arguments to BIO tagged masks
        labels.append((triple_to_bio_tags(ann, 0, lookup),
                       triple_to_bio_tags(ann, 1, lookup),
                       triple_to_bio_tags(ann, 2, lookup)))

        # Flatten turn sequence
        tokens.append([t for ts in ann['tokens'] for t in ts + ['<eos>']])

    # Show as BIO scheme
    i = random.randint(0, len(tokens) - 1)
    df = pd.DataFrame(labels[i], columns=tokens[i], index=['subj', 'pred', 'obj'])
    print(df)

    i = random.randint(0, len(labels))
    print(' '.join(tokens[i]) + '\n')

    print('Subjects:')
    print(bio_tags_to_tokens(['+' + t for t in tokens[i]], labels[i][0], bio_lookup))

    print('\nPredicates:')
    print(bio_tags_to_tokens(['+' + t for t in tokens[i]], labels[i][1], bio_lookup, predicate=True))

    print('\nObjects:')
    print(bio_tags_to_tokens(['+' + t for t in tokens[i]], labels[i][2], bio_lookup))




