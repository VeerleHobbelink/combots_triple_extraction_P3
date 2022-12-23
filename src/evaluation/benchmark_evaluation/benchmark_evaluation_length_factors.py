from src.model_transformer.run_transformer_pipeline import AlbertTripleExtractor
from metrics import recall_at_k
from nltk import word_tokenize
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import spacy


def load_examples(path):
    """ Load examples in the form of (str: dialogue, list: triples).

    :param path: Path to test file (e.g. 'test_examples/test_full.txt')
    :return:     List of (str: dialogue, list: triples) pairs.
    """
    # Extract lines corresponding to dialogs and triples
    samples = []
    with open(path, 'r', encoding='utf-8') as file:
        block = []
        for line in file:
            if line.strip() and not line.startswith('#'):
                block.append(line.strip())
            elif block:
                samples.append(block)
                block = []
        samples.append(block)

    # Split triple arguments
    examples = []
    for block in samples:
        dialog = block[1]
        triples = [string_to_triple(triple) for triple in block[2:]]
        examples.append((dialog, triples))
    return examples


def string_to_triple(text_triple):
    """ Tokenizes triple line into individual arguments

    :param text_triple: plain-text string of triple
    :return:            triple of the form (subj, pred, obj, polar)
    """
    return tuple([x.strip() for x in text_triple.split(',')])


def lemmatize_triple(subj, pred, obj, polar, nlp):
    """ Takes in a triple and perspective and lemmatizes/normalizes the predicate.
    """
    pred = ' '.join([t.lemma_ for t in nlp(pred)])
    return subj, pred, obj, polar


def evaluate(test_file, model, num_samples=-1, k=0.9, deduplication=True):
    """ Evaluates the model on a test file, yielding scores for precision@k,
        recall@k, F1@k and PR-AUC.

    :param test_file:     Test file from '/test_examples'
    :param model:         Albert, Dependency or baseline model instance
    :param num_samples:   The maximum number of samples to evaluate (default: all)
    :param k:             Confidence level at which to evaluate models
    :param deduplication: Whether to lemmatize predicates to make sure duplicate predicates such as "is"
                          and "are" are removed and match across baselines (default: True)
    :return:              Scores for precision@k, recall@k, F1@k and PR-AUC
    """
    # Extract dialog-triples pairs from annotations
    examples = load_examples(test_file)
    if num_samples > 0:
        examples = examples[:num_samples]

    # Predictions
    true_triples = []
    pred_triples = []
    dialog_lengths = []
    for i, (dialog, triples) in enumerate(examples):
        # Print progress
        print('\n (%s/%s) input: %s' % (i + 1, len(examples), dialog))

        # Make sure we have at least 3 turns
        num_eos = dialog.count('<eos>')
        if num_eos < 3:
            dialog = '<eos>' * (3 - num_eos) + dialog

        # Predict triples
        extractions = list(model.extract_triples(dialog, verbose=True))

        # Check for error in test set formatting
        error = False
        for triple in triples:
            if len(triple) != 4:
                print('#######\nERROR\n#######')
                print(triple)
                error = True

        print('expected:', triples)
        print('found:   ', [t for c, t in extractions if c > k])

        if not error:
            true_triples.append(triples)
            pred_triples.append(extractions)
            dialog_lengths.append(len(word_tokenize(dialog)))

    # If deduplication is enabled, map word forms to lemmas
    if deduplication:
        print('\nPerforming de-duplication')
        nlp = spacy.load('en_core_web_sm')
        true_triples = [set([lemmatize_triple(*triple, nlp) for triple in lst]) for lst in true_triples]
        pred_triples = [set([(c, lemmatize_triple(*triple, nlp)) for c, triple in lst]) for lst in pred_triples]

    # For each length, compute recall score
    lengths = sorted(list(set(dialog_lengths)))
    recalls = []
    for l in lengths:
        true_triples_length = [t for i, t in enumerate(true_triples) if dialog_lengths[i] == l]
        pred_triples_length = [t for i, t in enumerate(pred_triples) if dialog_lengths[i] == l]

        recall = recall_at_k(true_triples_length, pred_triples_length, k=k)
        recalls.append(recall)

    # Create a CSV to store plot into
    df = pd.DataFrame(list(zip(lengths, recalls)), columns=['length', 'recall'])
    df.to_csv('results/' + Path(test_file).stem + '_' + model.name + '_length_recall.csv')

    # Compute recall
    plt.plot(lengths, recalls)
    plt.xlabel('Length (tokens)')
    plt.ylabel('Recall')
    plt.show()


if __name__ == '__main__':
    MODEL = 'albert'

    bio_lookup = {3: 'like', 5: 'do'}  # change

    if MODEL == 'albert':
        model = AlbertTripleExtractor('../../model_transformer/models/2022-04-27', bio_lookup, speaker1='speaker1', speaker2='speaker2')
    else:
        raise Exception('model %s not recognized' % MODEL)

    evaluate('test_examples/test_single_utterances.txt', model, k=.2, num_samples=10)
