import numpy as np
import pandas as pd
import warnings
import gc
import os
import pickle
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from lightgbm import LGBMRegressor
import optuna
from scipy.stats import entropy, hmean, gmean
from scipy.signal import find_peaks
from sklearn.model_selection import KFold


class DifferencePredictor:
    """
    A class for predicting differences in composite scores between time periods.
    
    This class handles the entire pipeline of loading data, preprocessing, feature engineering,
    model optimization, and prediction for score differences. It uses LightGBM as the base model
    and includes options for TabPFN integration.

    Parameters
    ----------
    path_data_processed : str, default='../data/processed/'
        Path to processed data directory
    path_data_raw : str, default='../data/raw/'
        Path to raw data directory
    path_inference_dataset : str, optional
        Path to inference dataset
    path_quantiles_tabpfn : str, default='../output/quantiles_tabpfn.pkl'
        Path to TabPFN quantiles file
    path_models : str, default='../models/'
        Path to save/load models
    path_output : str, default='../output/'
        Path to save outputs
    n_trials : int, default=1
        Number of Optuna optimization trials
    n_seeds : int, default=1
        Number of random seeds for model training
    random_state : int, default=42
        Random seed for reproducibility
    include_tabpfn : bool, default=True
        Whether to include TabPFN features
    load_results : bool, default=False
        Whether to load existing results instead of recomputing

    Attributes
    ----------
    features_data : pd.DataFrame
        Loaded feature data
    train_labels : pd.DataFrame
        Training labels
    submission_format : pd.DataFrame
        Submission format template
    tabpfn_results : pd.DataFrame
        TabPFN prediction results
    """
    
    def __init__(
        self,
        path_data_processed='../data/processed/',
        path_data_raw='../data/raw/',
        path_inference_dataset=None,
        path_quantiles_tabpfn='../output/quantiles_tabpfn.pkl',
        path_models='../models/',
        path_output='../output/',
        n_trials=1,
        n_seeds=1,
        random_state=42,
        include_tabpfn=True,
        load_results=False
    ):
        """
        Initializes the ModelPipeline with specified configurations.

        Parameters:
        - processed_dir (str): Directory where processed data files are located/saved.
        - raw_dir (str): Directory where raw data files are located.
        - quantiles_tabpfn_path (str): Path to the quantiles_tabpfn pickle file.
        - n_trials (int): Number of trials for Optuna optimization.
        - n_seeds (int): Number of seeds for model training.
        - random_state (int): Random state for reproducibility.
        - include_tabpfn (bool): Whether to include TabPFN features.
        """
        # Suppress warnings
        warnings.filterwarnings("ignore")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        # Enable garbage collection
        gc.enable()
        
        # Directories
        self.path_data_processed = path_data_processed
        self.path_data_raw = path_data_raw
        self.path_inference_dataset = path_inference_dataset
        self.path_quantiles_tabpfn = path_quantiles_tabpfn
        self.path_models = path_models
        self.path_output = path_output

        # Hyperparameters
        self.n_trials = n_trials
        self.n_seeds = n_seeds
        self.random_state = random_state
        
        # Initialize data attributes
        self.features_data = None
        self.train_labels = None
        self.submission_format = None
        self.tabpfn_results = None
        
        self.full_train = None
        self.full_test = None
        self.train_features_all = None
        self.train_labels_all_wide = None
        self.train_features_all = None
        
        self.X_train_diff = None
        self.diff_features = None
        self.Y_train_diff = None
        
        self.tabpfn_results_wide = None
        
        self.study_diff = None
        self.y_pred_diff_train_fin = None
        self.y_pred_diff_test_fin = None
        
        self.include_tabpfn = include_tabpfn
        self.load_results = load_results
        
    def load_data(self):
        """
        Load all necessary datasets from specified directories.
        
        Loads:
        - features_data from processed directory
        - train_labels from raw directory
        - submission_format from raw directory
        - tabpfn_results from quantiles file
        
        Raises
        ------
        Exception
            If any of the required files cannot be loaded
        """
        try:
            self.features_data = pd.read_pickle(f'{self.path_data_processed}features_data.pkl')
            self.train_labels = pd.read_csv(f'{self.path_data_raw}train_labels.csv')
            self.submission_format = pd.read_csv(f'{self.path_data_raw}submission_format.csv')
            self.tabpfn_results = pd.read_pickle(self.path_quantiles_tabpfn)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def merge_and_preprocess(self):
        """
        Merge and preprocess training and testing data.
        
        This method:
        1. Merges training labels with features
        2. Merges submission format with features for test data
        3. Samples based on year participation
        4. Converts labels to wide format
        5. Computes differences between years
        6. Prepares training and test feature sets
        7. Handles TabPFN integration if enabled
        
        The processed data is stored in class attributes for further use.
        """

        # Merge and preprocess training data
        self.full_train = (
            self.train_labels.reset_index(drop=True)
            .merge(self.features_data, on='uid', how='left')
            .drop(columns=['composite_score'], errors='ignore')
            .reset_index(drop=True)
        )
        # Save full_train
        self.full_train.to_pickle(f'{self.path_data_processed}full_train.pkl')
        
        # Merge and preprocess testing data
        self.full_test = (
            self.submission_format
            .merge(self.features_data, on='uid', how='left')
            .drop(columns=['composite_score'], errors='ignore')
            .reset_index(drop=True)
        )
        # Save full_test
        self.full_test.to_pickle(f'{self.path_data_processed}full_test.pkl')
        
        # Sampling based on year participation
        uids_train = self.train_labels.groupby('uid').filter(lambda x: len(x) == 2)['uid'].unique()
        self.train_features_all = self.full_train[self.full_train['uid'].isin(uids_train)].reset_index(drop=True)
        
        # Convert labels to wide format and compute differences
        self.train_labels_all_wide = self.pivot_and_diff(self.train_labels, ['composite_score_16', 'composite_score_21'])
        
        # Remove duplicates and drop irrelevant columns
        self.train_features_all = self.train_features_all.drop_duplicates(subset=['uid']).drop(columns=['year', 'feature_category'], errors='ignore').reset_index(drop=True)
        
        # Merge features with wide labels
        self.train_features_all = self.train_features_all.merge(self.train_labels_all_wide, on='uid', how='left')
        
        # Drop intermediate columns
        stats_cols = ['composite_score_16', 'composite_score_21', 'composite_score_diff']
        self.train_features_all.drop(columns=stats_cols, inplace=True, errors='ignore')
        
        # Split into training and test data
        self.Y_train_diff = self.train_labels_all_wide[['uid', 'composite_score_diff']].dropna().reset_index(drop=True)
        self.X_train_diff = self.train_features_all.loc[self.train_features_all['uid'].isin(self.Y_train_diff['uid'])].reset_index(drop=True)
        
        # With NAS - full diff
        self.Y_train_diff_full = self.train_labels_all_wide[['uid', 'composite_score_diff']].reset_index(drop=True)
        self.X_train_diff_full = self.full_train.drop_duplicates(subset=['uid']).drop(columns=['year', 'feature_category'], errors='ignore').reset_index(drop=True)
        self.X_test_diff_full = self.full_test.drop_duplicates(subset=['uid']).drop(columns=['year', 'feature_category'], errors='ignore').reset_index(drop=True)
        
        # Out of sample dfs that do not require nested CV
        self.diff_features = pd.concat([self.X_train_diff_full, self.X_test_diff_full], axis=0).reset_index(drop=True)
        self.diff_features = self.diff_features.loc[~self.diff_features['uid'].isin(self.Y_train_diff['uid'])].reset_index(drop=True)
        self.diff_features_uid = self.diff_features['uid']
        self.diff_features = self.diff_features.drop(columns=['uid', 'feature_category'], errors='ignore')
        
        # Target
        self.Y_train_diff = self.Y_train_diff['composite_score_diff']
        
        # Convert uid to numeric and clean column names
        for df in [self.X_train_diff, self.diff_features]:
            df.columns = df.columns.str.replace(r'[^\w]', '_', regex=True)
        
        # Drop uid column
        self.X_train_diff_uid = self.X_train_diff['uid']
        self.X_train_diff = self.X_train_diff.drop(columns=['uid', 'feature_category'], errors='ignore')
        
        # Tabpfn_results to wide based on year - only if include_tabpfn is True
        if self.include_tabpfn:

            ll_cols = self.tabpfn_results.drop(columns=['year', 'uid'], axis=1, errors='ignore').columns
            self.tabpfn_results_wide = self.tabpfn_results.pivot(index='uid', columns='year', values=ll_cols).reset_index()
            
            # Rename columns
            self.tabpfn_results_wide.columns = ['_'.join(map(str, col)).strip() for col in self.tabpfn_results_wide.columns.values]
            self.tabpfn_results_wide = self.tabpfn_results_wide.loc[:, ~self.tabpfn_results_wide.columns.duplicated()]
            self.tabpfn_results_wide = self.tabpfn_results_wide.rename(columns={'uid_': 'uid'})
            
            # Merge tabpfn_results_wide with training and diff_features
            self.X_train_diff = pd.concat([self.X_train_diff, self.X_train_diff_uid], axis=1).merge(
                self.tabpfn_results_wide, on=['uid'], how='left'
            ).drop(columns=['uid', 'feature_category'], axis=1, errors='ignore')
            self.diff_features = pd.concat([self.diff_features, self.diff_features_uid], axis=1).merge(
                self.tabpfn_results_wide, on=['uid'], how='left'
            ).drop(columns=['uid', 'feature_category'], axis=1, errors='ignore')
        
    def pivot_and_diff(self, df, columns):
        """
        Pivot the dataframe and compute differences between time periods.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data to pivot
        columns : list
            List of column names for the pivoted data

        Returns
        -------
        pd.DataFrame
            Pivoted DataFrame with additional difference column calculated as:
            (first_period - second_period) / second_period
        """
        wide_df = df.pivot(index='uid', columns='year', values='composite_score').reset_index()
        wide_df.columns = ['uid'] + columns
        wide_df['composite_score_diff'] = (wide_df[columns[0]] - wide_df[columns[1]]) / wide_df[columns[1]]
        return wide_df
    
    def apply_log_transformation(self):
        """
        Apply log transformation to specified numeric columns.
        
        Transforms income-related columns using natural logarithm.
        Handles zero or negative values by replacing them with NaN.
        
        The transformation is applied to both training and feature datasets.
        
        Raises
        ------
        Exception
            If any error occurs during the transformation process
        """
        try:
            columns_to_log = [
                'hincome_03', 'hincome_12', 'hinc_business_03', 'hinc_business_12',
                'hinc_rent_03', 'hinc_rent_12', 'hinc_assets_03', 'hinc_assets_12',
                'hinc_cap_03', 'hinc_cap_12', 'rinc_pension_03', 'rinc_pension_12',
                'sinc_pension_03', 'sinc_pension_12',
                'hincome_diff','hinc_business_diff',
                'hinc_rent_diff', 'hinc_assets_diff'
            ]
            
            for col in columns_to_log:
                if col in self.X_train_diff.columns and col in self.diff_features.columns:
                    self.X_train_diff[col] = np.where(self.X_train_diff[col] > 0, np.log(self.X_train_diff[col]), np.nan)
                    self.diff_features[col] = np.where(self.diff_features[col] > 0, np.log(self.diff_features[col]), np.nan)
                else:
                    print(f"Column '{col}' not found in both X_train_diff and diff_features. Skipping log transformation for this column.")
        except Exception as e:
            print(f"Error during log transformation: {e}")
            raise
    
    def add_features(self):
        """
        Add interaction and polynomial features.
        
        Creates new features:
        - age_squared: squared value of narrowed_age_12
        - age_income_interaction: interaction between age and income
        
        The features are added to both training and feature datasets.
        
        Raises
        ------
        Exception
            If required columns are not found in the datasets
        """
        try:
            # Add interaction and polynomial features for age
            for df in [self.X_train_diff, self.diff_features]:
                if 'narrowed_age_12' in df.columns and 'hincome_12' in df.columns:
                    df['age_squared'] = df['narrowed_age_12'] ** 2
                    df['age_income_interaction'] = df['narrowed_age_12'] * df['hincome_12']
                else:
                    print("Warning: 'narrowed_age_12' or 'hincome_12' not found in dataframe. Skipping feature addition.")
        except Exception as e:
            print(f"Error adding features: {e}")
            raise
    
    def advanced_stats(self, df, group_vars):
        """
        Calculate advanced statistical measures for grouped data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the data
        group_vars : list
            List of columns to group by

        Returns
        -------
        pd.DataFrame
            DataFrame containing calculated statistics including:
            - Basic statistics (min, max, mean, std, skew, median)
            - Advanced measures (range, IQR, MAD, CV, Gini)
            - Distribution statistics (entropy, harmonic mean, geometric mean)
            - Signal analysis (peaks, troughs)
            - Quintile values
        """
        # Calculate standard aggregations separately
        grouped_stats = df.groupby(group_vars)['composite_score_diff'].agg([
            'min', 'max', 'mean', 'std', 'skew', 'median'
        ]).reset_index()
        
        # Custom aggregations - apply separately
        grouped_stats['range'] = df.groupby(group_vars)['composite_score_diff'].apply(lambda x: x.max() - x.min()).values
        grouped_stats['iqr'] = df.groupby(group_vars)['composite_score_diff'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).values
        grouped_stats['mad'] = df.groupby(group_vars)['composite_score_diff'].apply(lambda x: (x - x.mean()).abs().mean()).values  # Manual calculation of MAD
        grouped_stats['cv'] = df.groupby(group_vars)['composite_score_diff'].apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0).values
        grouped_stats['gini'] = df.groupby(group_vars)['composite_score_diff'].apply(
            lambda x: sum([abs(i - j) for i in x for j in x]) / (2 * len(x) * sum(x)) if sum(x) != 0 else 0
        ).values
        grouped_stats['entropy'] = df.groupby(group_vars)['composite_score_diff'].apply(
            lambda x: entropy(pd.Series(x).value_counts(normalize=True))
        ).values
        grouped_stats['harmonic_mean'] = df.groupby(group_vars)['composite_score_diff'].apply(
            lambda x: hmean(x) if all(x > 0) else 0
        ).values
        grouped_stats['geometric_mean'] = df.groupby(group_vars)['composite_score_diff'].apply(
            lambda x: gmean(x) if all(x > 0) else 0
        ).values
        grouped_stats['peaks'] = df.groupby(group_vars)['composite_score_diff'].apply(
            lambda x: len(find_peaks(x)[0])
        ).values
        grouped_stats['troughs'] = df.groupby(group_vars)['composite_score_diff'].apply(
            lambda x: len(find_peaks(-x)[0])
        ).values
    
        # Calculate quintiles separately
        quintile_labels = ['q1_20', 'q2_40', 'q3_60', 'q4_80']
        quintiles_df = df.groupby(group_vars)['composite_score_diff'].apply(
            lambda x: pd.Series(x.quantile([0.2, 0.4, 0.6, 0.8]).values)
        ).unstack()
        quintiles_df.columns = quintile_labels
        quintiles_df.reset_index(inplace=True)
    
        # Merge quintiles with grouped stats
        grouped_stats = grouped_stats.merge(quintiles_df, on=group_vars, how='left')
        
        return grouped_stats
    
    def compute_advanced_statistics(self):
        """
        Compute advanced statistical features based on various grouping variables.
        
        This method:
        1. Groups data by education, age, and year participation
        2. Calculates advanced statistics for each group
        3. Merges results back to training and feature datasets
        
        The computed statistics include distribution measures, variability metrics,
        and percentile-based features.
        """
        try:
            # Prepare composite score with the grouping variables for easier grouping
            df = pd.concat([self.X_train_diff, self.Y_train_diff], axis=1)
            
            # Define potential grouping variables
            potential_group_vars = [
                ['edu_gru_12'],
                ['narrowed_age_12'],
                ['year_participation']
            ]
            
            # Filter to only use variables that exist in the dataframe
            valid_group_vars = [vars for vars in potential_group_vars 
                              if all(var in df.columns for var in vars)]
            
            for group_var in valid_group_vars:
                grouped_stats = self.advanced_stats(df, group_var)
                
                # Flatten the list for the suffix in column names
                suffix = '_'.join(group_var)
                self.X_train_diff = self.X_train_diff.merge(
                    grouped_stats, on=group_var, how='left', suffixes=('', f'_{suffix}')
                )
                self.diff_features = self.diff_features.merge(
                    grouped_stats, on=group_var, how='left', suffixes=('', f'_{suffix}')
                )
            
        except Exception as e:
            print(f"Error computing advanced statistics: {e}")
            raise
    
    def optimize_hyperparameters(self):
        """
        Optimize LightGBM hyperparameters using Optuna.
        
        If load_results is True and a saved study exists, loads the previous study.
        Otherwise, performs optimization with cross-validation to find optimal parameters.
        
        The optimization process includes:
        1. Defining parameter search spaces
        2. K-fold cross-validation for each trial
        3. RMSE minimization
        4. Saving the optimized study
        """
        try:
            study_path = f'{self.path_models}study_diff.pkl'
            
            # Try to load existing study only if load_results is True
            if self.load_results and os.path.exists(study_path):
                print("Loading existing study...")
                with open(study_path, 'rb') as f:
                    self.study_diff = pickle.load(f)
                return

            # Define the objective function for Optuna
            def objective(trial):
                # Suggest hyperparameters
                params = {
                    'objective': 'regression_l2',  # L2 loss function
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 15, 50),  # Increased range for more flexibility
                    'max_depth': trial.suggest_int('max_depth', 8, 10),  # Moderate depth to balance overfitting
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),  # Lower learning rate
                    'n_estimators': trial.suggest_int('n_estimators', 50, 100),  # Number of trees
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Introduce randomness for smoother predictions
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Introduce more feature randomness
                    'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 10.0),  # Stronger L1 regularization
                    'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 10.0),  # Stronger L2 regularization
                    'random_state': 1
                }
                # Set seed in params
                params['seed'] = params['random_state']
                
                # Initialize LGBMRegressor with suggested parameters
                model = LGBMRegressor(**params, verbose=-1)
                
                # K-Fold cross-validation
                kf = KFold(n_splits=5, random_state=42, shuffle=True)
                rmse_scores = []
                
                for train_index, val_index in kf.split(self.X_train_diff):
                    X_train_fold, X_val_fold = self.X_train_diff.iloc[train_index], self.X_train_diff.iloc[val_index]
                    y_train_fold, y_val_fold = self.Y_train_diff.iloc[train_index], self.Y_train_diff.iloc[val_index]
                    
                    # Train model on this fold
                    model.fit(X_train_fold, y_train_fold)
                    
                    # Predict and evaluate RMSE
                    y_pred = model.predict(X_val_fold)
                    rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                    rmse_scores.append(rmse)
                
                # Return the average RMSE across all folds
                return np.mean(rmse_scores)
            
            # Create and optimize the study
            self.study_diff = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5),
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            self.study_diff.optimize(objective, n_trials=self.n_trials, n_jobs=1)
            
            # Create models directory if it doesn't exist
            os.makedirs(self.path_models, exist_ok=True)
            
            # Save the study
            with open(study_path, 'wb') as f:
                pickle.dump(self.study_diff, f)
            
        except Exception as e:
            print(f"Error during hyperparameter optimization: {e}")
            raise
    
    def cross_validate_and_predict(self):
        """
        Perform cross-validation, model training, and prediction.
        
        If load_results is True and saved results exist:
        1. Loads existing model and predictions
        2. Only generates new test predictions
        
        Otherwise:
        1. Performs 5-fold cross-validation
        2. Trains final model on full dataset
        3. Generates and saves both training and test predictions
        
        Results are saved to specified output paths.
        """
        try:
            model_path = f'{self.path_models}model_diff.pkl'
            train_pred_path = f'{self.path_output}pred_diff_train.pkl'
            
            if self.load_results:
                # Load existing model and predictions if available
                if os.path.exists(model_path):
                    print("Loading existing model...")
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    if os.path.exists(train_pred_path):
                        print("Loading existing training predictions...")
                        self.y_pred_diff_train_fin = pd.read_pickle(train_pred_path)
                        
                    # Only predict on test data
                    # Difference between sets of features in diff_features and features in model
                    diff_features_cols = set(self.diff_features.columns) - set(model.feature_name_)
                    print(diff_features_cols)

                    y_pred_diff_full = model.predict(self.diff_features)
                    y_pred_diff_full = pd.DataFrame(y_pred_diff_full, columns=['composite_score_diff'])
                    y_pred_diff_full['uid'] = self.diff_features_uid
                    
                    # Get only test predictions
                    self.y_pred_diff_test_fin = y_pred_diff_full[
                        y_pred_diff_full['uid'].isin(self.full_test['uid'])
                    ].reset_index(drop=True)
                    
                    # Save test predictions
                    self.y_pred_diff_test_fin.to_pickle(f'{self.path_output}pred_diff_test.pkl')
                    return
            
            # If not loading results or files don't exist, proceed with full training
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_diff_train = np.zeros_like(self.Y_train_diff.values)
            fold_numbers = np.zeros(self.Y_train_diff.shape[0], dtype=int)
            
            for fold, (train_index, val_index) in enumerate(cv.split(self.X_train_diff), start=1):
                X_tr, X_val = self.X_train_diff.iloc[train_index], self.X_train_diff.iloc[val_index]
                y_tr, y_val = self.Y_train_diff.iloc[train_index], self.Y_train_diff.iloc[val_index]
                
                # Train model using best parameters
                model = LGBMRegressor(**self.study_diff.best_params, verbose=-1, random_state=42)
                model.fit(X_tr, y_tr)
                
                y_pred_val = model.predict(X_val)
                y_pred_diff_train[val_index] = y_pred_val
                fold_numbers[val_index] = fold
            
            # Calculate final CV RMSE
            final_rmse = root_mean_squared_error(self.Y_train_diff, y_pred_diff_train)
            print("Difference CV RMSE:", final_rmse, "\n")
            
            # Train final model on entire training set
            model = LGBMRegressor(**self.study_diff.best_params, verbose=-1, random_state=42)
            model.fit(self.X_train_diff, self.Y_train_diff)
            
            # Save the trained model
            os.makedirs(self.path_models, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create training predictions DataFrame
            y_pred_diff_train = pd.DataFrame(y_pred_diff_train, columns=['composite_score_diff'])
            y_pred_diff_train['uid'] = self.X_train_diff_uid
            y_pred_diff_train['fold1'] = fold_numbers
            
            # Predict on full dataset
            y_pred_diff_full = model.predict(self.diff_features)
            y_pred_diff_full = pd.DataFrame(y_pred_diff_full, columns=['composite_score_diff'])
            y_pred_diff_full['uid'] = self.diff_features_uid
            
            # Separate train and test predictions
            y_pred_diff_full_train = y_pred_diff_full[
                y_pred_diff_full['uid'].isin(self.full_train['uid'])
            ].reset_index(drop=True)
            y_pred_diff_test_fin = y_pred_diff_full[
                y_pred_diff_full['uid'].isin(self.full_test['uid'])
            ].reset_index(drop=True)
            
            # Combine predictions and save
            self.y_pred_diff_train_fin = pd.concat([y_pred_diff_train, y_pred_diff_full_train], axis=0).reset_index(drop=True)
            self.y_pred_diff_test_fin = y_pred_diff_test_fin
            
            self.y_pred_diff_train_fin.to_pickle(train_pred_path)
            self.y_pred_diff_test_fin.to_pickle(f'{self.path_output}pred_diff_test.pkl')

        except Exception as e:
            print(f"Error during difference cross-validation and prediction: {e}")
            raise
    
    def process(self):
        """
        Execute the complete difference prediction pipeline.
        
        The pipeline includes:
        1. Loading and preprocessing data
        2. Feature engineering and transformation
        3. Computing advanced statistics
        4. Hyperparameter optimization
        5. Model training and prediction
        
        All intermediate results are saved to specified paths.
        """
        self.load_data()
        self.merge_and_preprocess()
        self.apply_log_transformation()
        self.add_features()
        self.compute_advanced_statistics()
        self.optimize_hyperparameters()
        self.cross_validate_and_predict()
