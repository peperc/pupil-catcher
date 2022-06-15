import pandas as pd
import numpy as np

from keyboard_map import KEYBOARD_MAP


EXPERIMENTS = [
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-IBK',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-J48',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-LMT',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-logistic',
    'D:/jrc/uc3m/tfm-repo/datasets/experiments/search-SVM'
]

COLUMNS = ['dataset', 'predict_means', 'predict_std', 'accuracy_means', 'accuracy_std']

def get_predictions_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=0)


def get_percentage_correct_and_accuracy(predictions: pd.DataFrame, dataset: str) -> pd.Series:
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
    
    return pd.DataFrame([[dataset, np.mean(folds_prediction_results), np.std(folds_prediction_results), np.mean(folds_accuracy_results), np.std(folds_accuracy_results)]], columns=COLUMNS)



if __name__ == '__main__':
    
    for experiment in EXPERIMENTS:  
        final_results = pd.DataFrame(columns=COLUMNS)
        
        for i in range(0,12,1):
            predictions = get_predictions_from_csv(experiment + f'/{str(i)}.csv')
            predictions['actual'] = predictions.apply(lambda x : x['actual'].split(':')[0], axis=1)
            predictions['predicted'] = predictions.apply(lambda x : x['predicted'].split(':')[0], axis=1)

            results = get_percentage_correct_and_accuracy(predictions, str(i))

            final_results = pd.concat([final_results, results], ignore_index=True)
        
        final_results.to_excel(experiment + '/results.xlsx', index=False)

        