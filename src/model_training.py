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

from scipy.stats import spearmanr
from utils import MultipleTimeSeriesCV, format_time
from rich import print
from rich.traceback import install

install()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_store', type=str, default='data_normalized/assets.h5',
                   help='Ruta al archivo H5 con los datos')
parser.add_argument('--data_store_item', type=str, default='engineered_features',
                   help='Nombre del dataset dentro del archivo H5')
parser.add_argument('--results_path', type=str, default='results/us_stocks',
                   help='Ruta donde guardar resultados')

args = parser.parse_args()

DATA_STORE = args.data_store
DATA_STORE_ITEM = args.data_store_item
RESULTS_PATH = Path(args.results_path)
LGB_STORE = Path(RESULTS_PATH / 'tuning_lgb.h5')


tiempo_inicial = time()

YEAR = 52
VAL_LENGTH = 9

TRAIN_VAL_END_YEAR = 2020

train_lengths = [464,]
test_lengths = [1, ]

# constraints on structure (depth) of each tree
MAX_DEPTHS = [2, 3, 5, 7]
NUM_LEAVES_OPTS = [2 ** i for i in MAX_DEPTHS]
MIN_DATA_IN_LEAF_OPTS = [250, 500, 1000]

# weight of each new tree in the ensemble
LEARNING_RATE_OPTS = [.01, .1, .3]

# random feature selection
FEATURE_FRACTION_OPTS = [.3, .6, .95]



if not RESULTS_PATH.exists():
    RESULTS_PATH.mkdir(parents=True)



def get_fi(model):
    """Return normalized feature importance as pd.Series"""
    fi = model.feature_importance(importance_type='gain')
    return (pd.Series(fi / fi.sum(),
                      index=model.feature_name()))

def ic_lgbm(preds, train_data):
    """Custom IC eval metric for lightgbm"""
    is_higher_better = True
    return 'ic', spearmanr(preds, train_data.get_label())[0], is_higher_better

def get_cv_params(cv_params, lookahead, train_length, test_length):
    n_params = len(cv_params)
    print(f'# Parameters: {n_params}')
    # randomized grid search
    cvp = np.random.choice(list(range(n_params)),
                           size=int(n_params / 2),
                           replace=False,
                           )
    cv_params_iteration = [cv_params[i] for i in cvp]

    # set up cross-validation
    n_splits = int(VAL_LENGTH * YEAR / test_length)

    # time-series cross-validation
    cv = MultipleTimeSeriesCV(n_splits=n_splits,
                              lookahead=lookahead,
                              test_period_length=test_length,
                              train_period_length=train_length)
    
    return cv_params_iteration, cv


data = (pd.read_hdf(DATA_STORE, DATA_STORE_ITEM) 
            .sort_index()
            .loc[pd.IndexSlice[:, :f'{TRAIN_VAL_END_YEAR - 1}'], :])

print("Tamaño de los datos:", data.shape)
print(data.head())


labels = sorted(data.filter(like='target_').columns)
features = data.columns.difference(labels).tolist()
# tickers = data.sector.unique()

lookaheads = [int(label.split('_')[-1].replace('w','')) for label in labels]

categoricals = [ 'sector']
for feature in categoricals:
    data[feature] = pd.factorize(data[feature], sort=True)[0]



label_dict = dict(zip(lookaheads, labels))
label_dict = {1:"target_1w"}
lookaheads = [1]


test_params = list(product(lookaheads, train_lengths, test_lengths))

base_params = dict(boosting='gbdt',
                   objective='regression',
                   verbose=-1,                   
                   )


param_names = ['learning_rate', 'num_leaves',
               'feature_fraction', 'min_data_in_leaf']

cv_params = list(product(LEARNING_RATE_OPTS,
                         NUM_LEAVES_OPTS,
                         FEATURE_FRACTION_OPTS,
                         MIN_DATA_IN_LEAF_OPTS))



num_iterations = [50, 75] + list(range(100, 501, 50))
num_boost_round = num_iterations[-1]

metric_cols = (param_names + ['t', 'daily_ic_mean', 'daily_ic_mean_n',
                              'daily_ic_median', 'daily_ic_median_n'] +
               [str(n) for n in num_iterations])

for lookahead, train_length, test_length in test_params:

    cv_params_iteration, cv = get_cv_params(cv_params, lookahead, train_length, test_length)

    print(f'Lookahead: {lookahead:2.0f} | '
          f'Train: {train_length:3.0f} | '
          f'Test: {test_length:2.0f} | '
          f'Params: {len(cv_params_iteration):3.0f} | '
          f'Train configs: {len(test_params)}')
    

    label = label_dict[lookahead]
    outcome_data = data.loc[:, features + [label]].dropna()
    
    # binary dataset
    lgb_data = lgb.Dataset(data=outcome_data.drop(label, axis=1),
                           label=outcome_data[label],
                           categorical_feature=categoricals,
                           free_raw_data=False)
    T = 0
    predictions, metrics, feature_importance, daily_ic = [], [], [], []
    
    # iterate over (shuffled) hyperparameter combinations
    for p, param_vals in enumerate(cv_params_iteration):
        key = f'{lookahead}/{train_length}/{test_length}/' + '/'.join([str(p) for p in param_vals])
        params = dict(zip(param_names, param_vals))
        params.update(base_params)

        start = time()
        cv_preds, nrounds = [], []
        ic_cv = defaultdict(list)
        
        # iterate over folds
        for i, (train_idx, test_idx) in tqdm(enumerate(cv.split(X=outcome_data))):            
            # select train subset
            lgb_train = lgb_data.subset(used_indices=train_idx.tolist(),
                                       params=params).construct()
            
            # train model for num_boost_round
            model = lgb.train(params=params,
                              train_set=lgb_train,
                              num_boost_round=num_boost_round                              
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
        
        # compute IC per day
        by_day = cv_preds.groupby(level='date')
        ic_by_day = pd.concat([by_day.apply(lambda x: spearmanr(x.y_test, x[str(n)])[0]).to_frame(n)
                               for n in num_iterations], axis=1)
        daily_ic_mean = ic_by_day.mean()
        daily_ic_mean_n = daily_ic_mean.idxmax()
        daily_ic_median = ic_by_day.median()
        daily_ic_median_n = daily_ic_median.idxmax()
        
        
        # compute IC across all predictions
        ic = [spearmanr(cv_preds.y_test, cv_preds[str(n)])[0] for n in num_iterations]
        t = time() - start
        T += t
        
        # collect metrics
        metrics = pd.Series(list(param_vals) +
                            [t, daily_ic_mean.max(), daily_ic_mean_n, daily_ic_median.max(), daily_ic_median_n] + ic,
                            index=metric_cols)
        msg = f'\t{p:3.0f} | {format_time(T)} ({t:3.0f}) | {params["learning_rate"]=:5.2f} | '
        msg += f'{params["num_leaves"]=:3.0f} | {params["feature_fraction"]=:3.0%} | {params["min_data_in_leaf"]=:4.0f} | '
        msg += f' {max(ic)=:6.2%} | {ic_by_day.mean().max()=: 6.2%} | {daily_ic_mean_n=: 4.0f} | {ic_by_day.median().max()=: 6.2%} | {daily_ic_median_n=: 4.0f}'
        print(msg)

        # persist results for given CV run and hyperparameter combination
        metrics.to_hdf(LGB_STORE, 'metrics/' + key)
        ic_by_day.assign(**params).to_hdf(LGB_STORE, 'daily_ic/' + key)
        fi.T.describe().T.assign(**params).to_hdf(LGB_STORE, 'fi/' + key)
        cv_preds.to_hdf(LGB_STORE, 'predictions/' + key)

tiempo_final = time()
print(f'Tiempo total de ejecución: {tiempo_final - tiempo_inicial} segundos')