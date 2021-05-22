from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def preprocessing(X, y, cv, params):
    '''Препроцессинг данных и параметров
    :param X: фичи для обучения модели
    :type X: numpy 2d array or pandas Dataframe
    :param y: таргет для обучения модели
    :type y: numpy array or pandas dataframe
    :param cv: индексы кросс валидации
    :type cv: iterable of (train_inex, test_index)
    :param params: функция, сэмплирующая параметры
    :type params: func
    :return: обработанный X, y (numpy arrays) и params
    :rtype: tuple
    '''

    if (type(X) == pd.core.frame.DataFrame) & ((type(y) == pd.core.frame.DataFrame)| (type(y) == pd.core.series.Series)):
        return X.values, y.values, cv, params
    else:
        return X, y, cv, params