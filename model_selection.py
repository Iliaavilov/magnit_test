import optuna
import numpy as np
import copy
import lightgbm as lgb
import preprocessing
import neptune
from neptunecontrib.monitoring.optuna import NeptuneCallback
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer


class training:
    '''Класс для подбора параметров модели и логгирования результатов
    '''
    def __init__(self,
                 direction = 'minimize', model = None,
                 params_func = None, n_trials = None,
                 random_state = None, sampler = None,
                 nn_training = None, nn_model = None):
        '''
        :param direction: направление оптимизации (minimize или maximize)
        :type direction: str
        :param model: какую модель оптимизировать (lgbm или torch)
        :type model: str
        :param params_func: функция, сэмплирующая параметры
        :type params_func: func
        :param n_trials: количество попыток оптимизации
        :type n_trials: int
        :param random_state: random seed
        :type random_state: int
        :param sampler: алгоритм для сэмплирования параметров
        :type sampler: optuna.samplers
        :param nn_training: класс, обучающий нейронную сеть
        :type nn_training: class
        :param nn_model: класс с нейронной сетью
        :type nn_model: class
        '''
        self.direction = direction
        self.model = model
        self.params_func = params_func
        self.n_trials = n_trials
        self.random_state = random_state
        self.sampler = sampler
        self.nn_training = nn_training
        self.nn_model = nn_model

    def initiate_neptune_exp(self, name = None, description = None,
                             params = None, properties = None,
                             tags = None,
                             upload_source_files = ('cv.py', 'model_selection.py', 'nn_model.py',
                                                    'nn_training.py', 'preprocessing.py')):
        '''Инициализация neptune.ai эксперимента
        :param name: название эксперимента
        :type name: str
        :param description: описание эксперимента
        :type description: str
        :param params: параметры эксперимента
        :type params: dictionary
        :param properties: свойства эксперимента
        :type properties: list
        :param tags: тэги эксперимента
        :type tags: list
        :param upload_source_files: файлы, которые необходимо залогировать в эксперименте
        :type upload_source_files: list
        :return: None
        :rtype: None
        '''
        self.neptune_parameters = {'name': name, 'description': description,
                                   'params': params, 'properties': properties,
                                   'tags': tags, 'upload_source_files': upload_source_files}

        neptune.create_experiment(name, description, params, properties, tags, upload_source_files)

    def train(self, X, y, cv):
        '''Подбор параметров модели
        :param X: фичи для обучения модели
        :type X: numpy 2d array or pandas dataframe
        :param y: таргет для обучения модели
        :type y: numpy array or pandas dataframe
        :param cv: индексы кросс валидации
        :type cv: iterable of (train_inex, test_index)
        :return: None
        :rtype: None
        '''

        self.study = optuna.create_study(sampler = self.sampler(seed = self.random_state),
                                         direction = self.direction)
        ## минимизируем ошибку
        self.study.optimize(lambda trial: self.objective(trial, X, y, cv, self.model, self.params_func),
                            n_trials = self.n_trials, callbacks=[NeptuneCallback()])

        neptune.stop()

    def lgbm_cv_test(self, X, y, cv_trans, params_trans):
        '''Тренировка lgbm модели на валидационной и тестовой части с заданными параметрами
        :param X: фичи для обучения модели
        :type X: numpy 2d array
        :param y: таргет для обучения модели
        :type y: numpy 1d array
        :param cv_trans: индексы кросс валидации
        :type cv_trans: iterable of (train_inex, test_index)
        :param params_trans: словарь параметров
        :type params_trans: dictionary
        :return: dictionary with keys 'loss_mean_cv' (mean loss on cross validation),
                                      'loss_std_cv' (std of loss on validation folds)
                                      'loss_test' (loss on test)
                                      'iterations': (number of iterations on validation folds)
        :rtype: dictionary
        '''

        validation_loss = params_trans['validation_loss']
        early_stopping_rounds = params_trans['early_stopping_rounds']
        categorical_feature = params_trans['categorical_feature']
        feature_name = params_trans['feature_name']
        params_trans.pop('validation_loss')
        params_trans.pop('early_stopping_rounds')
        params_trans.pop('categorical_feature')
        params_trans.pop('feature_name')
        def lgb_scoring(y_hat, data, validation_loss = validation_loss, is_higher_better = self.direction):
            '''Подсчет loss для early stopping в lgbm
            :param y_hat: предсказанный таргет
            :type y_hat: array
            :param data: датасет, на котором обучается модель
            :type data: Lightgbm.Dataset
            :param validation_loss: функция, возвращающая лосс (принимает первым аргументом y_true - реальный таргет,
                                                                          вторым аргуметром y_hat - предсказанный таргет)
            :type validation_loss: func
            :param is_higher_better: Направление оптимизации (maximize or minimize)
            :type is_higher_better: str
            :return: 'val_loss' (названия столбца с функцией потерь),
                      validation_loss(y_true, y_hat) (значение функции потерь)
                      True/False - максизировать или минимизировать validation_loss

            :rtype: tuple
            '''
            y_true = data.get_label()

            if is_higher_better == 'maximize':
                is_higher_better = True
            else:
                is_higher_better = False
            return 'val_loss', validation_loss(y_true, y_hat), is_higher_better

        y = y.reshape(-1, 1)

        train_data = lgb.Dataset(X, y, feature_name = feature_name, categorical_feature = categorical_feature)
        cv_model = lgb.cv(params = params_trans,
                          train_set = train_data,
                          folds = cv_trans[:-1],
                          feval = lgb_scoring,
                          early_stopping_rounds = early_stopping_rounds,
                          verbose_eval = False,
                          return_cvbooster = True)

        X_train = X[cv_trans[-1][0], :]
        y_train = y[cv_trans[-1][0]]
        X_test = X[cv_trans[-1][1], :]
        y_test = y[cv_trans[-1][1]]
        train_data = lgb.Dataset(X_train,
                                 y_train, feature_name = feature_name, categorical_feature = categorical_feature)
        test_data = lgb.Dataset(X_test,
                                y_test, feature_name = feature_name, categorical_feature = categorical_feature)
        evals_result = {}
        params_trans['n_estimators'] = round(len(cv_model['val_loss-mean']))
        test_model = lgb.train(params = params_trans,
                               train_set = train_data,
                               valid_sets = [test_data],
                               valid_names = ['test_data'],
                               feval = lgb_scoring,
                               evals_result = evals_result,
                               verbose_eval = False)

        test_loss = evals_result['test_data']['val_loss'][-1]
        return {'loss_mean_cv': cv_model['val_loss-mean'][-1],
                'loss_std_cv':cv_model['val_loss-stdv'][-1],
                'loss_test': test_loss,
                'iterations': len(cv_model['val_loss-mean'])}

    def sklearn_cv_test(self, X, y, cv_trans, params_trans):
        '''Тренировка sklearn модели на валидационной и тестовой части с заданными параметрами
        :param X: фичи для обучения модели
        :type X: numpy 2d array
        :param y: таргет для обучения модели
        :type y: numpy 1d array
        :param cv_trans: индексы кросс валидации
        :type cv_trans: iterable of (train_inex, test_index)
        :param params_trans: словарь параметров
        :type params_trans: dictionary
        :return: dictionary with keys 'loss_mean_cv' (mean loss on cross validation),
                                      'loss_std_cv' (std of loss on validation folds)
                                      'loss_test' (loss on test)
        :rtype: dictionary
        '''

        estimator = params_trans['estimator']
        validation_loss = params_trans['validation_loss']
        params_trans.pop('estimator')
        params_trans.pop('validation_loss')

        def sklearn_scoring(validation_loss = validation_loss, is_higher_better = self.direction):
            '''Подсчет loss для early stopping в lgbm
            :param validation_loss: функция, возвращающая лосс (принимает первым аргументом y_true - реальный таргет,
                                                                          вторым аргуметром y_hat - предсказанный таргет)
            :type validation_loss: func
            :return: sklearn.metrics.make_scorer - функция, возвращающая score в sklearn cross_validate
            :rtype: sklearn.metrics.make_scorer
            '''

            if is_higher_better == 'maximize':
                is_higher_better = True
            else:
                is_higher_better = False

            return make_scorer(validation_loss, greater_is_better = is_higher_better)

        scoring = sklearn_scoring(validation_loss = validation_loss, is_higher_better = self.direction)

        y = y.reshape(-1, 1)
        estimator_initialised = estimator(**params_trans)
        scores = cross_validate(estimator_initialised, X, y, scoring = scoring)


        return {'loss_mean_cv': np.mean(scores['test_score'][:-1]*(-1)),
                'loss_std_cv': np.std(scores['test_score'][:-1]),
                'loss_test': scores['test_score'][-1]*(-1)}


    def nn_cv_test(self, X, y, cv_trans, params_trans):
        '''Тренировка NN модели на валидационной и тестовой части с заданными параметрами
        :param X: фичи для обучения модели
        :type X: numpy 2d array
        :param y: таргет для обучения модели
        :type y: numpy 2d array of shape (-1, 1)
        :param cv_trans: индексы кросс валидации
        :type cv_trans: iterable of (train_inex, test_index)
        :param params_trans: словарь параметров
        :type params_trans: dictionary
        :return: dictionary with keys 'loss_mean_cv' (mean loss on cross validation),
                                      'loss_std_cv' (std of loss on validation folds)
                                      'loss_test' (loss on test)
                                      'iterations': (mean number of iterations on validation folds)
        :rtype: dictionary
        '''

        best_iters = []
        best_cv = []

        for fold in cv_trans[:-1]:
            nn_training_initialised = self.nn_training(self.nn_model, X, y)
            val_losses = nn_training_initialised.train(params_trans = params_trans, val_fold = True)
            best_iters.append(len(val_losses))
            best_cv.append(val_losses[-1])

        neptune.log_metric('std_cv_loss', np.std(best_cv))
        neptune.log_metric('mean_cv_loss', np.mean(best_cv))
        neptune.log_metric('iterations', np.mean(best_iters))

        mean_best_iter = round(np.mean(best_iters))
        nn_training_initialised  = self.training_nn(self.nn_model, X[cv_trans[-1][1], :], y[cv_trans[-1][1], :])
        test_losses = nn_training_initialised.train(params_trans = params_trans, val_fold = False)
        test_loss = test_losses[-1]


    def objective(self, trial, X, y, cv, model, params_func):
        '''
        :param trial: Класс подбора параметров
        :type trial: optuna.trial.Trial
        :param X: фичи для обучения модели
        :type X: numpy 2d array or pandas Dataframe
        :param y: таргет для обучения модели
        :type y: numpy array or pandas dataframe
        :param cv: индексы кросс валидации
        :type cv: iterable of (train_inex, test_index)
        :param model: какую модель оптимизировать (lgbm или torch)
        :type model: str
        :param params_func: функция, сэмплирующая параметры
        :type params_func: func
        :return: средний loss на валидационной части
        :rtype: float
        '''

        ## Множество параметров моделей
        params = params_func(trial, X)
        neptune.log_text('logged_params', str(params))

        X_trans, y_trans, cv_trans, params_trans = preprocessing.preprocessing(X.copy(),
                                                                               y.copy(),
                                                                               copy.deepcopy(cv),
                                                                               copy.deepcopy(params))

        if model == 'lgbm':
            results_dict = self.lgbm_cv_test(X_trans,
                                             y_trans,
                                             cv_trans,
                                             params_trans)

            neptune.log_metric('metric_mean_cv', results_dict['loss_mean_cv'])
            neptune.log_metric('metric_test', results_dict['loss_std_cv'])
            neptune.log_metric('metric_std_cv', results_dict['loss_test'])
            neptune.log_metric('iterations', results_dict['iterations'])

        if model == 'sklearn':
            results_dict = self.sklearn_cv_test(X_trans,
                                                y_trans,
                                                cv_trans,
                                                params_trans)

            neptune.log_metric('metric_mean_cv', results_dict['loss_mean_cv'])
            neptune.log_metric('metric_test', results_dict['loss_std_cv'])
            neptune.log_metric('metric_std_cv', results_dict['loss_test'])


        if model == 'torch':

            results_dict = self.nn_cv_test(X_trans,
                                            y_trans,
                                            cv_trans,
                                            params_trans)

            neptune.log_metric('metric_mean_cv', results_dict['loss_mean_cv'])
            neptune.log_metric('metric_test', results_dict['loss_std_cv'])
            neptune.log_metric('metric_std_cv', results_dict['loss_test'])
            neptune.log_metric('iterations', results_dict['iterations'])

        return results_dict['loss_mean_cv']