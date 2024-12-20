import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import sys, os
from time import time
from tqdm import tqdm

from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from utils import MultipleTimeSeriesCV, format_time

import optuna

YEAR = 52
VAL_LENGTH = 4


TRAIN_VAL_END_YEAR = 2020


def objective(trial):
    base_params = dict(boosting='gbdt',
                   objective='regression',
                   verbose=-1)

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 2**7),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.3, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 250, 1000)
    }
    key = f'{lookahead}/{train_length}/{test_length}/' + '/'.join([str(p) for p in params.values()])
    params.update(base_params)

    start = time()
    cv_preds, nrounds = [], []
    ic_cv = defaultdict(list)
    
    n_splits = int(VAL_LENGTH * YEAR / test_length)

    cv = MultipleTimeSeriesCV(n_splits=n_splits,
                              lookahead=lookahead,
                              test_period_length=test_length,
                              train_period_length=train_length)
    # iterate over folds
    for i, (train_idx, test_idx) in enumerate(cv.split(X=outcome_data)):
        
        # select train subset
        lgb_train = lgb_data.subset(used_indices=train_idx.tolist(),
                                    params=params).construct()
        
        # train model for num_boost_round
        model = lgb.train(params=params,
                            train_set=lgb_train,
                            num_boost_round=num_boost_round,
                            tree_learner='feature',
                            )
        # log feature importance
        if i == 0:
            fi = get_fi(model).to_frame()
        else:
            fi[i] = get_fi(model)

        # capture predictions
        test_set = outcome_data.iloc[test_idx, :]
        X_test = test_set.loc[:, model.feature_name()]
        y_test = test_set.loc[:, label]
        y_pred = {str(n): model.predict(X_test, num_iteration=n) for n in num_iterations}
        
        # record predictions for each fold
        cv_preds.append(y_test.to_frame('y_test').assign(**y_pred).assign(i=i))
    
    # combine fold results
    cv_preds = pd.concat(cv_preds).assign(**params)
    predictions.append(cv_preds)
    
    # # compute IC per day
    # by_day = cv_preds.groupby(level='date')
    # ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x.y_test, x[str(n)])[0]).to_frame(n)
    #                         for n in num_iterations], axis=1)
    # daily_ic_mean = ic_by_day.mean()
    # daily_ic_mean_n = daily_ic_mean.idxmax()
    # daily_ic_median = ic_by_day.median()
    # daily_ic_median_n = daily_ic_median.idxmax()
    
    
    # compute IC across all predictions
    ic = [spearmanr(cv_preds.y_test, cv_preds[str(n)])[0] for n in num_iterations]
    t = time() - start
    T += t

    return ic
    
    # # collect metrics
    # metrics = pd.Series(list(param_vals) +
    #                     [t, daily_ic_mean.max(), daily_ic_mean_n, daily_ic_median.max(), daily_ic_median_n] + ic,
    #                     index=metric_cols)
    
    # msg = f'\t{p:3.0f} | {format_time(T)} ({t:3.0f}) | {params["learning_rate"]:5.2f} | '
    # msg += f'{params["num_leaves"]:3.0f} | {params["feature_fraction"]:3.0%} | {params["min_data_in_leaf"]:4.0f} | '
    # msg += f' {max(ic):6.2%} | {ic_by_day.mean().max(): 6.2%} | {daily_ic_mean_n: 4.0f} | {ic_by_day.median().max(): 6.2%} | {daily_ic_median_n: 4.0f}'
    # print(msg)

    # persist results for given CV run and hyperparameter combination
    # metrics.to_hdf(lgb_store, 'metrics/' + key)
    # ic_by_day.assign(**params).to_hdf(lgb_store, 'daily_ic/' + key)
    # fi.T.describe().T.assign(**params).to_hdf(lgb_store, 'fi/' + key)
    # cv_preds.to_hdf(lgb_store, 'predictions/' + key)

data = (pd.read_hdf('data/assets.h5', 'engineered_features')
            .sort_index()
            .loc[pd.IndexSlice[:, :f'{TRAIN_VAL_END_YEAR}'], :])


labels = sorted(data.filter(like='target_').columns)
features = data.columns.difference(labels).tolist()
tickers = data.sector.unique()

# lookaheads = [int(label.split('_')[-1].replace('w','')) for label in labels]
labels = ["target_1w"]
lookaheads = [1]

categoricals = ['year', 'month', 'sector']
for feature in categoricals:
    data[feature] = pd.factorize(data[feature], sort=True)[0]


train_lengths = [int(10 * 52), 52 * 5]
test_lengths = [2 * 52, 52]

test_params = list(product(lookaheads, train_lengths, test_lengths))


results_path = Path('results', 'us_stocks')
if not results_path.exists():
    results_path.mkdir(parents=True)



label_dict = dict(zip(lookaheads, labels))
num_iterations = [10, 25, 50, 75] + list(range(100, 501, 50))
num_boost_round = num_iterations[-1]

metric_cols = (param_names + ['t', 'daily_ic_mean', 'daily_ic_mean_n',
                              'daily_ic_median', 'daily_ic_median_n'] +
               [str(n) for n in num_iterations])

# for lookahead, train_length, test_length in test_params:
#     # randomized grid search
#     cvp = np.random.choice(list(range(n_params)),
#                            size=int(n_params / 2),
#                            replace=False)
#     cv_params_ = [cv_params[i] for i in cvp]

#     # set up cross-validation
#     n_splits = int(VAL_LENGTH * YEAR / test_length)
#     print(f'Lookahead: {lookahead:2.0f} | '
#           f'Train: {train_length:3.0f} | '
#           f'Test: {test_length:2.0f} | '
#           f'Params: {len(cv_params_):3.0f} | '
#           f'Train configs: {len(test_params)}')

#     # time-series cross-validation
#     cv = MultipleTimeSeriesCV(n_splits=n_splits,
#                               lookahead=lookahead,
#                               test_period_length=test_length,
#                               train_period_length=train_length)

#     label = label_dict[lookahead]
#     outcome_data = data.loc[:, features + [label]].dropna()
    
#     # binary dataset
#     lgb_data = lgb.Dataset(data=outcome_data.drop(label, axis=1),
#                            label=outcome_data[label],
#                            categorical_feature=categoricals,
#                            free_raw_data=False)
#     T = 0
#     predictions, metrics, feature_importance, daily_ic = [], [], [], []
#     header = "\tIter | Time (Iter) | LR | Leaves | Feat % | Min Data | IC Max | IC Mean | Best N | IC Med | Best N"
#     print(header)
#     # iterate over (shuffled) hyperparameter combinations
#     for p, param_vals in enumerate(cv_params_):



        