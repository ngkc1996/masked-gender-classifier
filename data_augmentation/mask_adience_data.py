import pandas as pd
import numpy as np
import cv2
import os
import math
import random
from mask import create_mask

np.random.seed(0)

current_dir = os.path.abspath(os.path.dirname(__file__))
folds_path = os.path.join(current_dir, 'adience', 'folds')
faces_path = os.path.join(current_dir, 'adience', 'faces')
results_path = os.path.join(current_dir, 'adience', 'results')
fold_names = [os.path.join(folds_path, f) for f in os.listdir(
    folds_path) if os.path.isfile(os.path.join(folds_path, f))]

genders = {'m': 'male', 'f': 'female'}


def train_validate_test_split(df, train_percent=.6, validate_percent=.2):
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test


def make_directories(directory, dataset_name='untitled'):

    # make the results directory
    if not os.path.exists(directory):
        os.mkdir(directory)

    # make directory for dataset name
    datasets_dir = os.path.join(directory, dataset_name)
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)

    # make directory for each gender
    for g in genders.values():
        gender_dir = os.path.join(directory, dataset_name, g)
        if not os.path.exists(gender_dir):
            os.mkdir(gender_dir)
    return


def mask_dataframe(df, n_samples=0, dataset_splits=None, random_masks=False):
    counts = {'success': 0, 'failed': 0}
    if dataset_splits is None:
        print("Error. Please provide appropriate data splits / genders for the mask data")
        return
    elif sum([d for d in dataset_splits.values()]) != 1:
        print("Error. Please ensure data splits add up to 1")
        return

    # create trackers for each gender of size t_size
    t_size = 10
    gender_tracker = [[k] * math.floor(v * t_size)
                      for k, v in dataset_splits.items()]
    gender_tracker = [y for x in gender_tracker for y in x]
    random.shuffle(gender_tracker)

    current_num = {}
    for g in genders.values():
        current_num[g] = 0

    # iterate through labels
    for label, item in df.iterrows():
        # limit number of items processed
        if sum([c for c in counts.values()]) == n_samples:
            break

        # sort file and create new savepath
        gender = genders.get(item['gender'])
        if gender is None:
            print('No gender assigned. Skipping.')
            counts['failed'] += 1
            continue

        dataset_name = gender_tracker[current_num[gender]]
        new_save_path = os.path.join(
            current_dir, results_path, dataset_name, gender)

        # get filename
        header = 'coarse_tilt_aligned_face'
        face_id = item['face_id']
        image = item['original_image']
        filename = '.'.join([header, str(face_id), image])

        # get file path
        face = os.path.join(
            faces_path, item['user_id'], filename)

        # validate file path
        if os.path.exists(face):
            print(f'Loading face from {face}')
        else:
            print(f'File {face} does not exist')
            continue

        # create mask - return True if success, False if failed
        if create_mask(image_path=face, save_path=new_save_path, random_masks=random_masks):
            # increment current num for gender if success
            current_num[gender] += 1
            if current_num[gender] >= t_size:
                current_num[gender] = 0
            counts['success'] += 1
        else:
            counts['failed'] += 1

    print(counts)


def main():
    dfs = []
    # read metadata from folds
    for f in fold_names:
        print(f'reading from {f}')
        dfs.append(pd.read_csv(f, sep="\t"))

    # load and shuffle metadata
    metadata = pd.concat([df for df in dfs])
    metadata = metadata.sample(frac=1)

    dataset_splits = {
        'test': 0.1,
        'train': 0.8,
        'validate': 0.1}

    # create directories
    for d in dataset_splits.keys():
        print(d)
        make_directories(os.path.join(current_dir, results_path), d)

    mask_dataframe(metadata, n_samples=None,
                   dataset_splits=dataset_splits, random_masks=True)


if __name__ == "__main__":
    main()
