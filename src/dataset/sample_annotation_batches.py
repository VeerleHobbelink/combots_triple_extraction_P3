import os
import json
import random

random.seed(2)


if __name__ == '__main__':
    SAMPLES_PER_BATCH = 200
    NUM_SHARED = 4
    START_SHARING = 40
    NUM_SHARED_SAMPLES = 30
    PILOT_BATCH = 20
    NUM_BATCHES = 20
    DATASET = 'batches/merged_trainval_unannotated.json'

    # Create directory to store batches into
    if not os.path.exists('batches'):
        os.mkdir('batches')

    # Load full dataset
    with open(DATASET, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Remove duplicates (just in case)
    used_ids = set()
    new_data = []
    for sample in data:
        if sample['id'] not in used_ids:
            new_data.append(sample)
        used_ids.add(sample['id'])
    data = new_data

    # Divide subset of N * K samples into K batches
    batches = []
    samples = random.sample(data, SAMPLES_PER_BATCH * NUM_BATCHES)
    for i in range(NUM_BATCHES):
        j = i * SAMPLES_PER_BATCH
        batch = samples[j:j + SAMPLES_PER_BATCH]
        batches.append(batch)

    # Save dialogs (with each 4 batches containing some overlap)
    used_ids = set()
    for i in range(0, NUM_BATCHES, NUM_SHARED):
        group_of_batches = batches[i:i + NUM_SHARED] # 4 batches

        # Determine samples that should be shared
        shared_batch = group_of_batches[0][START_SHARING:START_SHARING + NUM_SHARED_SAMPLES]  # share middle 30 of first batch

        for j, batch in enumerate(group_of_batches):
            # Add shared batch to current batch
            batch[START_SHARING:START_SHARING + NUM_SHARED_SAMPLES] = shared_batch

            # Register samples
            for sample in batch:
                used_ids.add(sample['id'])

            # Save to file
            with open('batches/batch_%s.json' % str(i + j), 'w') as file1:
                json.dump(batch, file1, indent=4)

    print("{} samples divided over {} batches".format(len(used_ids), NUM_BATCHES))


    # Create batch will all remaining samples
    remaining_batch = [sample for sample in data if sample['id'] not in used_ids]
    random.shuffle(remaining_batch)

    pilot_batch = []
    if len(remaining_batch) > PILOT_BATCH:
        pilot_batch = remaining_batch[:PILOT_BATCH]
        remaining_batch = remaining_batch[PILOT_BATCH:]
    print("{} in pilot batch".format(len(pilot_batch)))
    print("{} remaining".format(len(remaining_batch)))

    with open('batches/pilot_batch.json', 'w') as file:
        json.dump(pilot_batch, file, indent=4)

    with open('batches/remaining_batch.json', 'w') as file:
        json.dump(remaining_batch, file, indent=4)

    # Some dataset statistics
    print('\n### Sampling statistics ###')
    dataset_balance = {'Circa':0, 'PersonaChat':0, 'DailyDialogs':0}
    for sample in data:
        if 'circa' in sample['id']:
            dataset_balance['Circa'] += 1
        if 'personachat' in sample['id']:
            dataset_balance['PersonaChat'] += 1
        if 'daily_dialogs' in sample['id']:
            dataset_balance['DailyDialogs'] += 1
    print(dataset_balance)


