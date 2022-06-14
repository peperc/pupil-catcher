import pandas as pd

from keyboard_map import KEYBOARD_MAP

EXPERIMENTS = [
    'D:\jrc\uc3m\tfm-repo\datasets\experiments\search-IBK',
    'D:\jrc\uc3m\tfm-repo\datasets\experiments\search-J48',
    'D:\jrc\uc3m\tfm-repo\datasets\experiments\search-LMT',
    'D:\jrc\uc3m\tfm-repo\datasets\experiments\search-logistic',
    'D:\jrc\uc3m\tfm-repo\datasets\experiments\search-SMO',
    'D:\jrc\uc3m\tfm-repo\datasets\experiments\search-SVM'
]


def get_predictions_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=0)


if __name__ == '__main__':
    for experiment in EXPERIMENTS:
        predictions = get_predictions_from_csv(PATH)
        predictions = predictions.apply(lambda x : x['actual'].split(':')[0], axis=1)
        predictions = predictions.apply(lambda x : x['predicted'].split(':')[0], axis=1)