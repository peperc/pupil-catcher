import pandas as pd
import numpy as np

from keyboard_map import KEYBOARD_MAP
from math import ceil


EXPERIMENTS = [
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-IBK',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-J48',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-LMT',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-logistic',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-SVM'
]

EXPERIMENTS_FORCED = [
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/forced',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/no_forced'
]

COLUMNS_1 = ['dataset', 'predict_means', 'predict_std', 'accuracy_means', 'accuracy_std']

COLUMNS_2 = ['dataset', 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

def get_predictions_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=0)

# Simple function to round numbers
def round_up(n, decimals=2):
    multiplier = 10 ** decimals
    return ceil(n * multiplier) / multiplier

def get_percentage_correct_and_accuracy(predictions: pd.DataFrame, dataset: int) -> pd.DataFrame:
    predictions = predictions.reset_index()  # make sure indexes pair with number of rows

    folds_prediction_results = []
    folds_accuracy_results = []
    
    correct_predictions = 0
    accumulative_accuracy = 0
    count = 0
    for index, row in predictions.iterrows():
        if row['inst#'] == 1 and index != 0:
            folds_prediction_results.append(correct_predictions*100/count)
            folds_accuracy_results.append(accumulative_accuracy/count)
            correct_predictions = 0
            accumulative_accuracy = 0
            count = 0
        
        if row['error'] != '+':
            correct_predictions += 1
        
        accuracy = KEYBOARD_MAP[int(row['actual'])-1][int(row['predicted'])-1]
        if accuracy == -1:
            accumulative_accuracy += KEYBOARD_MAP[int(row['predicted'])-1][int(row['actual'])-1]
        else:
            accumulative_accuracy += accuracy
        
        count += 1
    
    return pd.DataFrame([[dataset, np.mean(folds_prediction_results), np.std(folds_prediction_results), np.mean(folds_accuracy_results), np.std(folds_accuracy_results)]], columns=COLUMNS_1)


def get_accuracy_distances(predictions: pd.DataFrame, dataset: int) -> pd.DataFrame:
    predictions = predictions.reset_index()  # make sure indexes pair with number of rows
    
    accuracy_distances = [dataset]
    accuracy_distances.extend([0 for i in range(14)])
    for index, row in predictions.iterrows():
        accuracy = KEYBOARD_MAP[int(row['actual'])-1][int(row['predicted'])-1]
        if accuracy == -1:
            accuracy = KEYBOARD_MAP[int(row['predicted'])-1][int(row['actual'])-1]    
        accuracy_distances[accuracy+1] += 1
    return pd.DataFrame([accuracy_distances])


def get_predictions_n(predictions: pd.DataFrame, dataset: int) -> pd.DataFrame:
    predictions = predictions.reset_index()  # make sure indexes pair with number of rows

    results_per_fold = [] # This will be the array of arrays, one row per fold, 14 columns per n predictions
    
    result_n = [0 for i in range(15)]
    count = 0
    for index, row in predictions.iterrows():
        if row['inst#'] == 1 and index != 0:
            results_per_fold.append([predictions*100/count for predictions in result_n])
            result_n = [0 for i in range(15)]
            count = 0
        
        accuracy = KEYBOARD_MAP[int(row['actual'])-1][int(row['predicted'])-1]
        if accuracy == -1:
            accuracy = KEYBOARD_MAP[int(row['predicted'])-1][int(row['actual'])-1]    
        
        for i in range(accuracy, 15, 1):
            result_n[i] += 1
        
        count += 1

    results_per_fold = np.array(results_per_fold)
    percentages = results_per_fold.mean(axis=0)
    stds = np.std(results_per_fold, axis=0)

    test_results = [dataset]
    test_results.extend([f'{round_up(percentages[i])} ({round_up(stds[i])})' for i in range(0,15,1)])
        
    return pd.DataFrame([test_results], columns=COLUMNS_2)


# Result extractor 1: Predictions and accuracy means and standard deviation
def pre_acc_mean_dev():
    for experiment in EXPERIMENTS:  
        final_results = pd.DataFrame(columns=COLUMNS_1)
        
        for i in range(0,12,1):
            predictions = get_predictions_from_csv(experiment + f'/{str(i)}.csv')
            predictions['actual'] = predictions.apply(lambda x : x['actual'].split(':')[0], axis=1)
            predictions['predicted'] = predictions.apply(lambda x : x['predicted'].split(':')[0], axis=1)

            results = get_percentage_correct_and_accuracy(predictions, i)

            final_results = pd.concat([final_results, results], ignore_index=True)
        
        final_results.to_excel(experiment + '/results.xlsx', index=False)


# Result extractor 2: Accuracy distances, note that the number of keystrokes will be multiplied by 10 because of the 10 folds
def acc_distances():
    for experiment in EXPERIMENTS:  
        final_results = pd.DataFrame()

        for i in range(0,12,1):
                predictions = get_predictions_from_csv(experiment + f'/{str(i)}.csv')
                predictions['actual'] = predictions.apply(lambda x : x['actual'].split(':')[0], axis=1)
                predictions['predicted'] = predictions.apply(lambda x : x['predicted'].split(':')[0], axis=1)

                results = get_accuracy_distances(predictions, i)

                final_results = pd.concat([final_results, results], ignore_index=True)
            
        final_results.to_excel(experiment + '/distances.xlsx', index=False)


# Result extractor 3: Get percentage of correct predictions but less restrictive. With error from 0 (perfection) to 14 (less restrictive, 100% of accuracy)
def pre_n(experiments: list[str], tests: int):
    for experiment in experiments:  
        final_results = pd.DataFrame(columns=COLUMNS_2)

        for i in range(0,tests,1):
                predictions = get_predictions_from_csv(experiment + f'/{str(i)}.csv')
                predictions['actual'] = predictions.apply(lambda x : x['actual'].split(':')[0], axis=1)
                predictions['predicted'] = predictions.apply(lambda x : x['predicted'].split(':')[0], axis=1)

                results = get_predictions_n(predictions, i)

                final_results = pd.concat([final_results, results], ignore_index=True)
            
        final_results.to_excel(experiment + '/predictions_n.xlsx', index=False)

if __name__ == '__main__':
    pre_n(EXPERIMENTS_FORCED, 10)
    