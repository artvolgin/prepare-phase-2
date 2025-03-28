# Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, average_precision_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import  KFold
import optuna
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import kurtosis, entropy, hmean, gmean
from scipy.signal import find_peaks
from pandas.api.types import CategoricalDtype
from sklearn.linear_model import Lasso
from scipy.special import softmax
import uuid
import gc
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
import random
import json
import os
import pickle
# suppress warnings
import warnings
warnings.filterwarnings("ignore")


class TabpfnColumnsSelector:
    """Class for selecting important columns using CatBoost feature importance."""

    def __init__(self,
                 path_data_raw = '../data/raw/',
                 importance_threshold = 0.6,
                 path_selected_columns = '../models/selected_columns_tabpfn.json',
                 path_inference_dataset = None,
                 path_models = '../models/',
                 path_output = '../output/',
                 load_results=False):
        """
        Initializes the TabpfnColumnsSelector with paths and parameters.

        Args:
            path_data_raw (str): Path to the raw data directory.
            importance_threshold (float): Threshold for feature importance.
            path_selected_columns (str): Path to save selected columns.
            path_inference_dataset (str, optional): Path to the inference dataset.
            path_models (str): Path to the models directory.
            path_output (str): Path to the output directory.
            load_results (bool): Whether to load existing results.
        """
        self.path_data_raw = path_data_raw
        self.importance_threshold = importance_threshold
        self.path_selected_columns = path_selected_columns
        self.load_results = load_results
        self.path_models = path_models
        self.path_inference_dataset = path_inference_dataset
        self.path_output = path_output
    @staticmethod
    def get_feature_importance_catboost(train, test, target, params, n_splits=5, verbose=0, early_stopping_rounds=20):
        """
        Computes feature importance using CatBoost.

        Args:
            train (pd.DataFrame): Training features.
            test (pd.DataFrame): Test features.
            target (pd.Series): Target variable.
            params (dict): Parameters for CatBoost.
            n_splits (int): Number of splits for KFold.
            verbose (int): Verbosity level.
            early_stopping_rounds (int): Early stopping rounds.

        Returns:
            pd.DataFrame: DataFrame containing feature importance.
        """
        oof = np.zeros(train.shape[0])
        predictions = np.zeros(test.shape[0])
        feature_importance_df = pd.DataFrame()
        
        kfold = KFold(n_splits=n_splits, random_state=42, shuffle=True)
        for fold, (trn_idx, val_idx) in enumerate(kfold.split(train)):
            x_train, y_train = train.iloc[trn_idx], target.iloc[trn_idx]
            x_valid, y_valid = train.iloc[val_idx], target.iloc[val_idx]
            model = CatBoostRegressor(**params)
            model.fit(x_train.drop(columns=['uid'], axis=1, errors='ignore'),
                    y_train, verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            oof[val_idx] = model.predict(x_valid.drop(columns=['uid'], axis=1, errors='ignore'))
            predictions += model.predict(test) / n_splits
            
            # feature importance
            fold_importance_df = pd.DataFrame()
            fold_importance_df["Feature"] = train.drop(columns=['uid'], axis=1, errors='ignore').columns
            fold_importance_df["importance"] = model.feature_importances_
            fold_importance_df["fold"] = fold + 1
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
            del x_train, y_train, x_valid, y_valid
            gc.collect()
        return feature_importance_df
    
    def get_selected_columns(self):
        """
        Selects columns based on feature importance and saves them.

        Returns:
            list: List of selected column names.
        """
        # Load all .csv files in the current directory
        test_features = pd.read_csv(self.path_inference_dataset)
        train_features = pd.read_csv(f'{self.path_data_raw}train_features.csv')
        train_labels = pd.read_csv(f'{self.path_data_raw}train_labels.csv')
        submission_format = pd.read_csv(f'{self.path_data_raw}submission_format.csv')

        # define categorical columns - if type - object
        cat_cols = train_features.select_dtypes(include='object').columns.tolist()
        train_features_catboost = train_features.copy()
        test_features_catboost = test_features.copy()

        train_features_catboost[cat_cols] = train_features_catboost[cat_cols].fillna('Nan')
        test_features_catboost[cat_cols] = test_features_catboost[cat_cols].fillna('Nan')

        # df for catboost
        train_features_catboost = train_labels.merge(train_features_catboost, on='uid', how='left').drop('composite_score', axis=1)
        test_features_catboost = submission_format.merge(test_features_catboost, on='uid', how='left').drop('composite_score', axis=1)

        # run the function
        params = {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_state': 42,
            'cat_features': cat_cols[1:]
        }

        feature_importance_df = self.get_feature_importance_catboost(train_features_catboost,
                                                                test_features_catboost.drop(columns=['uid'], axis=1, errors='ignore'),
                                                                train_labels['composite_score'], params)
        feature_importance_df = feature_importance_df.groupby('Feature').mean().reset_index().sort_values('importance', ascending=False).drop('fold', axis=1).reset_index(drop=True)

        # Filter columns for dtype=object or binary (exactly 2 unique values)
        filtered_cols = [
            col for col in train_features_catboost.columns 
            if  col.find('inc') == -1 and col.find('hrs') == -1 and col.find('earnings') == -1 and col.find('job_end') == -1 and col != 'uid'
        ]

        # filtered_cols = train_features_catboost.columns 
        train_features_catboost[filtered_cols] = train_features_catboost[filtered_cols].astype(str)

        # Prepare a list to hold the results
        value_counts_list = []

        # Compute value_counts for each relevant column
        for col in filtered_cols:
            counts = train_features_catboost[col].value_counts()
            for category, count in counts.items():
                value_counts_list.append({'Feature': col, 'Categories': category, 'N': count})

        # Convert the list of dictionaries into a DataFrame
        value_counts_df = pd.DataFrame(value_counts_list)

        # Display the resulting DataFrame
        feature_table = feature_importance_df.merge(value_counts_df, on='Feature', how='left')
        feature_table = feature_table[['Feature', 'Categories', 'N', 'importance']]
        feature_table['importance'] = np.round(feature_table['importance'], 3)

        selected_columns = feature_table[feature_table['importance'] > self.importance_threshold]['Feature'].unique()
        
        # Convert selected_columns to a list to ensure it's JSON serializable
        selected_columns = selected_columns.tolist()

        # save selected columns
        with open(self.path_selected_columns, 'w') as f:
            json.dump(selected_columns, f)
        print(f'Selected columns saved to {self.path_selected_columns}')

        return selected_columns


class TabpfnPredictor:
    """Class for predicting quantiles using TabPFN."""

    def __init__(self,
                 n_ens_configs=128,
                 quantiles_n=10,
                 kfold_n=5,
                 path_data_raw = '../data/raw/',
                 path_selected_columns = '../models/selected_columns_tabpfn.json',
                 path_inference_dataset = None,
                 path_output = '../output/',
                 path_models = '../models/',
                 load_results=False):
        """
        Initializes the TabpfnPredictor with paths and parameters.

        Args:
            n_ens_configs (int): Number of ensemble configurations.
            quantiles_n (int): Number of quantiles.
            kfold_n (int): Number of KFold splits.
            path_data_raw (str): Path to the raw data directory.
            path_selected_columns (str): Path to selected columns.
            path_inference_dataset (str, optional): Path to the inference dataset.
            path_output (str): Path to the output directory.
            path_models (str): Path to the models directory.
            load_results (bool): Whether to load existing results.
        """
        self.n_ens_configs = n_ens_configs
        self.quantiles_n = quantiles_n
        self.kfold_n = kfold_n
        self.path_data_raw = path_data_raw
        self.path_selected_columns = path_selected_columns
        self.path_output = path_output
        self.load_results = load_results
        self.path_inference_dataset = path_inference_dataset
        self.path_models = path_models
    
    def get_quantiles(self, train_features, train_labels, test_features, submission_format, quantiles_n, kfold_n, n_ens_configs):
        """
        Computes quantiles using TabPFN.

        Args:
            train_features (pd.DataFrame): Training features.
            train_labels (pd.DataFrame): Training labels.
            test_features (pd.DataFrame): Test features.
            submission_format (pd.DataFrame): Submission format.
            quantiles_n (int): Number of quantiles.
            kfold_n (int): Number of KFold splits.
            n_ens_configs (int): Number of ensemble configurations.

        Returns:
            pd.DataFrame: DataFrame containing quantile predictions.
        """
        random.seed(42)
        np.random.seed(42)

        # # Split y into 5 classes
        y_train = pd.qcut(train_labels['composite_score'], quantiles_n, labels=False)

        if os.path.exists(f'{self.path_output}tabpfn_oof_predictions.pkl') and self.load_results:
            oof_predictions = pd.read_pickle(f'{self.path_output}tabpfn_oof_predictions.pkl')
            print(f'Loaded oof_predictions from {self.path_output}tabpfn_oof_predictions.pkl')
        else:
            # Prepare data for 5-fold cross-validation
            kf = KFold(n_splits=kfold_n, shuffle=True, random_state=42)
            # DataFrames to store predictions
            oof_predictions = np.zeros((len(train_features), quantiles_n))
            # Perform 5-fold cross-validation
            for train_index, val_index in kf.split(train_features):
                # Split the data
                X_tr, X_val = train_features.iloc[train_index], train_features.iloc[val_index]
                y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
                # Initialize the classifier
                classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=n_ens_configs)
                # Train the model
                classifier.fit(X_tr, y_tr, overwrite_warning=True)
                # Predict probabilities for validation data
                val_proba = classifier.predict_proba(X_val)
                oof_predictions[val_index] = val_proba 

            # Aggregate out-of-fold predictions
            oof_predictions = pd.DataFrame(oof_predictions, columns=[f'quantile_{i}_{quantiles_n}' for i in range(quantiles_n)])
            oof_predictions['uid'] = train_labels['uid'].values
            oof_predictions['year'] = train_labels['year'].values
            # Save oof_df to output folder
            oof_predictions.to_pickle(f'{self.path_output}tabpfn_oof_predictions.pkl')

        # Predict probabilities for test data
        if os.path.exists(f'{self.path_models}tabpfn_classifier.pkl') and self.load_results:
            # classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=n_ens_configs)
            classifier = pickle.load(open(f'{self.path_models}tabpfn_classifier.pkl', 'rb'))
            print(f'Loaded classifier from {self.path_models}tabpfn_classifier.pkl')
        else:
            classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=n_ens_configs)
            classifier.fit(train_features, y_train, overwrite_warning=True)
            # Save classifier to output folder
            pickle.dump(classifier, open(f'{self.path_models}tabpfn_classifier.pkl', 'wb'))
            print(f'Saved classifier to {self.path_models}tabpfn_classifier.pkl')
        
        p_eval = classifier.predict_proba(test_features)
        df_test_quantiles = pd.DataFrame(p_eval)
        df_test_quantiles.columns = [f'quantile_{i}_{quantiles_n}' for i in range(quantiles_n)]
        df_test_quantiles['uid'] = submission_format['uid'].values
        df_test_quantiles['year'] = submission_format['year'].values
        df_quantiles = pd.concat([oof_predictions, df_test_quantiles], axis=0)

        return df_quantiles
    
    def process(self, selected_columns):
        """
        Processes the data and computes quantiles.

        Args:
            selected_columns (list): List of selected columns.

        Returns:
            pd.DataFrame: DataFrame containing quantile predictions.
        """
        # load data
        test_features = pd.read_csv(self.path_inference_dataset)
        train_features = pd.read_csv(f'{self.path_data_raw}train_features.csv')
        train_labels = pd.read_csv(f'{self.path_data_raw}train_labels.csv')
        submission_format = pd.read_csv(f'{self.path_data_raw}submission_format.csv')

        # prepare data
        train_features = train_features.merge(train_labels, on='uid', how='left')
        test_features = test_features.merge(submission_format, on='uid', how='left')
        train_features = train_features[list(selected_columns) + ['composite_score', 'uid']]
        test_features = test_features[list(selected_columns) + ['uid']]
        train_features = pd.get_dummies(train_features.drop(columns=['uid']))
        test_features = pd.get_dummies(test_features.drop(columns=['uid']))
        test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

        train_features.columns = train_features.columns.str.replace(r"[^\w]", "_", regex=True)
        test_features.columns = test_features.columns.str.replace(r"[^\w]", "_", regex=True)
        train_features = train_features.drop(columns=['composite_score'], axis=1, errors='ignore')
        test_features = test_features.drop(columns=['composite_score'], axis=1, errors='ignore')

        df_quantiles = self.get_quantiles(train_features, train_labels, test_features, submission_format,
                                          quantiles_n=self.quantiles_n, kfold_n=self.kfold_n, n_ens_configs=self.n_ens_configs)
        
        return df_quantiles
    

class TabpfnRunner:
    """Class for running the TabPFN process."""

    def __init__(self,
                 path_data_raw = '../data/raw/',
                 path_selected_columns = '../models/selected_columns_tabpfn.json',
                 path_quantiles_tabpfn = '../output/quantiles_tabpfn.pkl',
                 path_inference_dataset = None,
                 n_ens_configs = 128,
                 quantiles_n = 10,
                 kfold_n = 5,
                 path_models = '../models/',
                 path_output = '../output/',
                 load_results=False):
        """
        Initializes the TabpfnRunner with paths and parameters.

        Args:
            path_data_raw (str): Path to the raw data directory.
            path_selected_columns (str): Path to selected columns.
            path_quantiles_tabpfn (str): Path to save quantile predictions.
            path_inference_dataset (str, optional): Path to the inference dataset.
            n_ens_configs (int): Number of ensemble configurations.
            quantiles_n (int): Number of quantiles.
            kfold_n (int): Number of KFold splits.
            path_models (str): Path to the models directory.
            path_output (str): Path to the output directory.
            load_results (bool): Whether to load existing results.
        """
        self.path_data_raw = path_data_raw
        self.path_selected_columns = path_selected_columns
        self.load_results = load_results
        self.path_quantiles_tabpfn = path_quantiles_tabpfn
        self.path_inference_dataset = path_inference_dataset
        self.n_ens_configs = n_ens_configs
        self.quantiles_n = quantiles_n
        self.kfold_n = kfold_n
        self.path_models = path_models
        self.path_output = path_output

    def process(self):
        """
        Processes the data, selects columns, and computes quantiles.
        """
        if os.path.exists(self.path_selected_columns) and self.load_results:
            with open(self.path_selected_columns, 'r') as f:
                selected_columns = json.load(f)
        else:
            tabpfn_columns_selector = TabpfnColumnsSelector(path_data_raw=self.path_data_raw,
                                                        path_inference_dataset=self.path_inference_dataset,
                                                        path_selected_columns=self.path_selected_columns,
                                                        load_results=self.load_results)
            selected_columns = tabpfn_columns_selector.get_selected_columns()

        tabpfn_predictor = TabpfnPredictor(path_data_raw=self.path_data_raw,
                                           path_inference_dataset=self.path_inference_dataset,
                                          path_selected_columns=self.path_selected_columns,
                                          load_results=self.load_results,
                                          n_ens_configs=self.n_ens_configs,
                                          quantiles_n=self.quantiles_n,
                                          kfold_n=self.kfold_n)
        df_quantiles = tabpfn_predictor.process(selected_columns)
        df_quantiles.to_pickle(self.path_quantiles_tabpfn)
        print(f'Quantiles saved to {self.path_quantiles_tabpfn}')

