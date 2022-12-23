import json
import glob
from pathlib import Path
from Levenshtein import distance as levenshtein_distance


def get_trainval_examples(path='trainval'):
    dialogs = []
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            tokens = json.load(file)['tokens']

            dialog = '<eos>'.join([' '.join([t for t in turn]) for turn in tokens])
            dialog = dialog.replace(' [unk]', '').strip().lower()
            dialogs.append(dialog)
    return dialogs


def get_test_examples(path):
    lines = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                lines.append(line.lower().strip())
    return lines


def intersection(lst1, lst2, k=10):
    # soft matching intersection
    overlapping_items = []
    for item1 in set(lst1):
        for item2 in set(lst2):
            if levenshtein_distance(item1, item2) < k:
                overlapping_items.append(item2)
                break
    return overlapping_items


if __name__ == '__main__':
    # Load training set examples
    trainval_examples = get_trainval_examples()
    
    # Compare to each test set file
    for test_fname in glob.glob('test/*.txt'):
        print('File:', Path(test_fname).stem)
        
        test_examples = get_test_examples(test_fname)

        overlap = intersection(trainval_examples, test_examples)
        if overlap:
            for item in overlap:
                print('\t-', item)
        else:
            print('\t- No overlap with train')
        
