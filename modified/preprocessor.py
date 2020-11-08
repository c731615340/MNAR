"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback".
"""

import codecs
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_datasets(data: str, seed: int = 0) -> Tuple:
    """Load and preprocess raw datasets (Yahoo! R3 or Coat)."""
    if data == 'yahoo':
        with codecs.open(f'../data/{data}/train.txt', 'r', 'utf-8', errors='ignore') as f:
            data_train = pd.read_csv(f, delimiter='\t', header=None)
            data_train.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
            counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            print("counts for yahoo train set:", counts)
            for index in data_train.index:
                if data_train.iloc[index][2] == 1:
                    counts['1'] += 1
                elif data_train.iloc[index][2] == 2:
                    counts['2'] += 1
                elif data_train.iloc[index][2] == 3:
                    counts['3'] += 1
                elif data_train.iloc[index][2] == 4:
                    counts['4'] += 1
                elif data_train.iloc[index][2] == 5:
                    counts['5'] += 1
            print("counts for yahoo train set:", counts)
        with codecs.open(f'../data/{data}/test.txt', 'r', 'utf-8', errors='ignore') as f:
            data_test = pd.read_csv(f, delimiter='\t', header=None)
            data_test.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
            counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            print("counts for yahoo test set:", counts)
            for index in data_test.index:
                if data_test.iloc[index][2] == 1:
                    counts['1'] += 1
                elif data_test.iloc[index][2] == 2:
                    counts['2'] += 1
                elif data_test.iloc[index][2] == 3:
                    counts['3'] += 1
                elif data_test.iloc[index][2] == 4:
                    counts['4'] += 1
                elif data_test.iloc[index][2] == 5:
                    counts['5'] += 1
            print("counts for yahoo test set:", counts)
        for _data in [data_train, data_test]:
            _data.user, _data.item = _data.user - 1, _data.item - 1

        
    elif data == 'coat':
        col = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
        with codecs.open(f'../data/{data}/train.ascii', 'r', 'utf-8', errors='ignore') as f:
            data_train = pd.read_csv(f, delimiter=' ', header=None)
            data_train = data_train.stack().reset_index().rename(columns=col)
            data_train = data_train[data_train.rate.values != 0].reset_index(drop=True)

            counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            print("counts for train set:", counts)
            for index in data_train.index:
                if data_train.iloc[index][2] == 1:
                    counts['1'] += 1
                elif data_train.iloc[index][2] == 2:
                    counts['2'] += 1
                elif data_train.iloc[index][2] == 3:
                    counts['3'] += 1
                elif data_train.iloc[index][2] == 4:
                    counts['4'] += 1
                elif data_train.iloc[index][2] == 5:
                    counts['5'] += 1
            print("counts for train set:", counts)
            # for index in data_train.index:
            #     if data_train.iloc[index][2] == 3:
            #         data_train.iloc[index][2] = 4
            #     elif data_train.iloc[index][2] == 2:
            #         data_train.iloc[index][2] = 1
            #     elif data_train.iloc[index][2] == 5:
            #         data_train.iloc[index][2] = 2
            #     elif data_train.iloc[index][2] == 4:
            #         data_train.iloc[index][2] = 3
            #     else:
            #        data_train.iloc[index][2] = 5
        with codecs.open(f'../data/{data}/test.ascii', 'r', 'utf-8', errors='ignore') as f:
            data_test = pd.read_csv(f, delimiter=' ', header=None)
            data_test = data_test.stack().reset_index().rename(columns=col)
            data_test = data_test[data_test.rate.values != 0].reset_index(drop=True)
            
            counts = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            print("counts for test set:", counts)
            for index in data_test.index:
                if data_test.iloc[index][2] == 1:
                    counts['1'] += 1
                elif data_test.iloc[index][2] == 2:
                    counts['2'] += 1
                elif data_test.iloc[index][2] == 3:
                    counts['3'] += 1
                elif data_test.iloc[index][2] == 4:
                    counts['4'] += 1
                elif data_test.iloc[index][2] == 5:
                    counts['5'] += 1
            print("counts for test set:", counts)
            # for index in data_test.index:
            #     if data_test.iloc[index][2] == 3:
            #         data_test.iloc[index][2] = 2
            #     elif data_test.iloc[index][2] == 2:
            #         data_test.iloc[index][2] = 3
                # elif data_test.iloc[index][2] == 5:
                #     data_test.iloc[index][2] = 2
                # elif data_test.iloc[index][2] == 5:
                #     data_test.iloc[index][2] = 1
                # else:
                #     print(data_test.iloc[index][2])
        #data_all = pd.concat([data_train, data_test], axis = 0)
        # #data_all = data_train.append(data_test)
        # print(data_all)
        # print(len(data_train.index))
        # print(len(data_test.index))
        # print(len(data_all.index))
    #data_train, data_test = train_test_split(data_all.values, test_size=0.1, random_state=seed)
    
    test = data_test.values
    train, val = train_test_split(data_train.values, test_size=0.1, random_state=seed)
    num_users, num_items = train[:, 0].max() + 1, train[:, 1].max() + 1

    return train, val, test, num_users, num_items
