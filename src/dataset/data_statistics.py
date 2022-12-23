import glob
import json
import numpy as np
import matplotlib.pyplot as plt


def load_annotations(path):
    annotations = []
    for fname in glob.glob(path + '/*.json'):
        with open(fname, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if not data['skipped']:
                data['tokens'] = [[t for t in turn if t != '[unk]'] for turn in data['tokens']]
                annotations.append(data)
    return annotations


def piechart(sizes, labels, explode_size=0.02):
    explode = np.zeros(len(sizes))
    explode[1] = explode_size

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


def show_stats(path):
    directories = glob.glob(path + '/*')
    print('Number of annotators:', len(directories))

    print('\nAnnotator contributions:')
    all_annotations = {}
    for dir_ in directories:
        annotations = load_annotations(dir_)
        all_annotations.update(annotations)
        print(dir_, ' N =', len(annotations))

    print('\nDataset contributions:')
    dataset_count = {'circa': 0, 'daily_dialogs': 0, 'personachat': 0}
    for fname in list(all_annotations.keys()):
        for dataset_name in dataset_count.keys():
            if dataset_name in fname:
                dataset_count[dataset_name] += 1
    for dataset_name, count in dataset_count.items():
        print(dataset_name, count)

    piechart((dataset_count['personachat'], dataset_count['daily_dialogs'], dataset_count['circa']),
             ('PersonaChat', 'DailyDialogs', 'Circa'))

    print('\nAggregate statistics:')
    dialogue_stats = {'tokens in turn': [], 'tokens in dialogue': [], 'triples in dialogue': []}

    for sample in all_annotations.values():
        # Count length of individual turns (in tokens)
        for turn in sample['tokens']:
            dialogue_stats['tokens in turn'].append(len(turn))

        # Count length of dialogue (in tokens)
        dialogue_stats['tokens in dialogue'].append(sum([len(t) for t in sample['tokens']]))

        # Count number of triples for dialogue
        num_triples = sum([any(triple) for triple in sample['annotations']])
        dialogue_stats['triples in dialogue'].append(num_triples)

    print('#Dialogs:', len(all_annotations))
    print('#Turns per dialog:', 3)
    print('#Turns:', len(dialogue_stats['tokens in turn']))
    print('#Tokens:', sum(dialogue_stats['tokens in turn']))
    print('#Triples:', sum(dialogue_stats['triples in dialogue']))
    print('#Triples per dialogue:', np.mean(dialogue_stats['triples in dialogue']))
    print()
    print('Avg. tokens per turn:', np.mean(dialogue_stats['tokens in turn']))
    print('Avg. tokens in dialogue:', np.mean(dialogue_stats['tokens in dialogue']))


if __name__ == '__main__':
    show_stats('final/trainval')
