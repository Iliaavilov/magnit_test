import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_indices_ts(data, splits, date_col):
    '''Time series cross validation
    :param data: датафрейм
    :type data: dataframe
    :param splits: список границ тестовой части валидационных фолдов
    :type splits: iterable of (validation_start, validation_end)
    :param date_col: колонка датафрейма, по которой необходимо делить по времени
    :type date_col: str
    :return: индексы кросс валидации
    :rtype: iterable of (train_inex, test_index)
    '''

    assert type(data.index) == pd.core.indexes.range.RangeIndex

    cv = []
    for split in splits:
        train = data[data[date_col]<pd.to_datetime(split[0])].index.values
        test = data[(data[date_col] >= pd.to_datetime(split[0])) &
                    (data[date_col] < pd.to_datetime(split[1]))].index.values
        cv.append((train, test))
    return(cv)



def get_indices_skf(X, y, n_splits, random_state):
    '''
    :param X: фичи для обучения модели
    :type X: numpy 2d array or pandas dataframe
    :param y: таргет для обучения модели
    :type y: numpy array or pandas dataframe
    :param n_splits: количество фолдов
    :type n_splits: int
    :param random_state: andom seed
    :type random_state: int
    :return:
    :rtype:
    '''


    skf = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle = True)
    cv = list(skf.split(X, y))
    return(cv)