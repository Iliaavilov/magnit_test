import pandas as pd

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