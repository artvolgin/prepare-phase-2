import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, average_precision_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
from sklearn.linear_model import Lasso
import xgboost as xgb
from scipy.special import softmax
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


class EnsemblePredictor:
    """
    A class to perform ensemble predictions using various regression models.
    
    Attributes:
    - n_trials: Number of trials for hyperparameter optimization.
    - n_seeds: Number of seeds for model training.
    - load_results: Boolean indicating whether to load existing results.
    - path_data_processed: Path to processed data.
    - path_data_raw: Path to raw data.
    - path_quantiles_tabpfn: Path to quantiles file.
    - path_output: Path to output directory.
    - path_models: Path to models directory.
    - path_inference_dataset: Path to inference dataset.
    - filename_results: Filename for results.
    """

    def __init__(self, n_trials, n_seeds, load_results,
                 path_data_processed='../data/processed/',
                 path_data_raw='../data/raw/',
                 path_quantiles_tabpfn='../output/quantiles_tabpfn.pkl',
                 path_output='../output/',
                 path_models='../models/',
                 path_inference_dataset=None,
                 filename_results='results.pkl'):
        """
        Initialize the EnsemblePredictor with paths and parameters.
        """
        self.n_trials = n_trials
        self.n_seeds = n_seeds
        self.load_results = load_results
        self.path_data_processed = path_data_processed
        self.path_data_raw = path_data_raw
        self.path_quantiles_tabpfn = path_quantiles_tabpfn
        self.path_output = path_output
        self.path_models = path_models
        self.path_inference_dataset = path_inference_dataset
        self.filename_results = filename_results

    def objective_lgbm(self, boosting_type, X_train, Y_train):
        """
        Define the objective function for LightGBM hyperparameter optimization.

        Parameters:
        - boosting_type: Type of boosting to use.
        - X_train: Training features.
        - Y_train: Training labels.

        Returns:
        - objective: A function to be used by Optuna for optimization.
        """
        def objective(trial):
            
            SEED = 42
            # Set seeds locally within each trial
            np.random.seed(SEED + trial.number)
        
            # Suggest hyperparameters
            params = {
                'objective': 'regression_l2',  # L2 loss function
                'metric': 'rmse',
                'boosting_type': boosting_type,
                'num_leaves': trial.suggest_int('num_leaves', 15, 50),
                'max_depth': trial.suggest_int('max_depth', 8, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 10.0),
                'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 10.0),
                'random_state': 1
            }
            # set seed in params
            params['seed'] = params['random_state']
            
            # Initialize LGBMRegressor with suggested parameters
            model = LGBMRegressor(**params, verbose=-1)
            
            # K-Fold cross-validation
            kf = GroupKFold(n_splits=5)
            #kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmse_scores = []
            
            for train_index, val_index in kf.split(X_train, Y_train, groups=X_train['combined_fold']):
            #for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]
                
                # Train model on this fold
                model.fit(
                    X_train_fold.drop(columns=['combined_fold', 'feature_category'], axis=1, errors='ignore'),
                    y_train_fold,
                    eval_set=[
                        (X_val_fold.drop(columns=['combined_fold', 'feature_category'], axis=1, errors='ignore'), y_val_fold)
                    ]
                )

                # Predict and evaluate RMSE
                y_pred = model.predict(X_val_fold.drop(columns=['combined_fold', 'feature_category'], axis=1, errors='ignore'))
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                rmse_scores.append(rmse)
            
            # Return the average RMSE across all folds
            return np.mean(rmse_scores)
        
        return objective

    def objective_catboost(self, metric, X_train, Y_train):
        """
        Define the objective function for CatBoost hyperparameter optimization.

        Parameters:
        - metric: Evaluation metric to optimize.
        - X_train: Training features.
        - Y_train: Training labels.

        Returns:
        - objective: A function to be used by Optuna for optimization.
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {
                'iterations': trial.suggest_int('iterations', 700, 1000),
                'depth': trial.suggest_int('depth', 2, 6),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.08, 0.2),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 4, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_strength': trial.suggest_float('random_strength', 0.5, 2),
                'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'verbose': 0,
                'random_state': 42
            }
            
            # Initialize CatBoost model with suggested parameters
            model = CatBoostRegressor(**params)
            
            # K-Fold cross-validation
            kf = GroupKFold(n_splits=5)
            scores = []
            
            for train_index, val_index in kf.split(X_train, Y_train, X_train['combined_fold']):
            #for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]
                
                # Train model on this fold
                model.fit(
                    X_train_fold.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'), y_train_fold, 
                    eval_set=(X_val_fold.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'), y_val_fold),
                    early_stopping_rounds=50
                )
                
                # Predict
                y_pred = model.predict(X_val_fold.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'))
                
                # Evaluate based on the selected metric
                if metric == 'rmse':
                    score = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                elif metric == 'mae':
                    score = mean_absolute_error(y_val_fold, y_pred)
                elif metric == 'map':
                    score = average_precision_score(y_val_fold, y_pred)
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                scores.append(score)
            
            # Return the average score across all folds
            # Negate the score if the metric needs to be maximized (e.g., MAP)
            if metric == 'map':
                return -np.mean(scores)
            return np.mean(scores)
        
        return objective

    def objective_xgboost(self, trial, X_train, Y_train):
        """
        Define the objective function for XGBoost hyperparameter optimization.

        Parameters:
        - trial: Optuna trial object.
        - X_train: Training features.
        - Y_train: Training labels.

        Returns:
        - Average RMSE across all folds.
        """
        #Suggest hyperparameters
        params = {
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                    "booster": "gbtree",
                    "lambda": trial.suggest_loguniform("lambda", 0.1, 10.0),
                    "alpha": trial.suggest_loguniform("alpha", 0.1, 10.0),
                    "eta": trial.suggest_loguniform("eta", 0.1, 0.25),
                    "gamma": trial.suggest_loguniform("gamma", 0.01, 10.0),
                    "max_depth": trial.suggest_int("max_depth", 3, 7),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
                    "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                    "n_estimators": trial.suggest_int("n_estimators", 80, 200),
                    'random_state': 42
            }
        # Initialize CatBoost model with suggested parameters
        model = xgb.XGBRegressor(**params)
        
        # K-Fold cross-validation
        kf = GroupKFold(n_splits=5)
        rmse_scores = []
        for train_index, val_index in kf.split(X_train, Y_train, X_train['combined_fold']):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]
            
            # Train model on this fold
            model.fit(X_train_fold.drop(columns = ['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'), y_train_fold, 
                    eval_set=[(X_val_fold.drop(columns = ['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'), y_val_fold)],
                    verbose=False)
            
            # Predict and evaluate RMSE
            y_pred = model.predict(X_val_fold.drop(columns = ['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'))
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            rmse_scores.append(rmse)
        
            # Return the average RMSE across all folds
            return np.mean(rmse_scores)

    def train_and_evaluate_model(self, X_train, Y_train, X_train_uid, X_test, X_test_uid, study, regressor_type, seed):
        """
        Train and evaluate a model with the specified regressor type and seed.

        Parameters:
        - X_train: Training features.
        - Y_train: Training labels.
        - X_train_uid: Unique IDs for training data.
        - X_test: Test features.
        - X_test_uid: Unique IDs for test data.
        - study: Optuna study object with best parameters.
        - regressor_type: Type of regressor to use.
        - seed: Random seed for reproducibility.

        Returns:
        - y_pred_train_df: DataFrame of training predictions.
        - y_pred_test_df: DataFrame of test predictions.
        - train_rmse: RMSE of training predictions.
        """
        # Define regressor based on regressor_type
        if regressor_type == 'lgbm':
            from lightgbm import LGBMRegressor
            Regressor = lambda **kwargs: LGBMRegressor(**kwargs, verbose=-1)
        elif regressor_type == 'catboost':
            from catboost import CatBoostRegressor
            Regressor = lambda **kwargs: CatBoostRegressor(**kwargs, verbose=0)
        elif regressor_type == 'xgboost':
            from xgboost import XGBRegressor
            Regressor = lambda **kwargs: XGBRegressor(**kwargs, verbose=0)
        else:
            raise ValueError("Unsupported regressor type.")

        # Initialize y_pred_train
        y_pred_train = np.zeros(len(Y_train), dtype=float)

        if (self.load_results and
            os.path.exists(f'{self.path_output}/y_pred_train_{regressor_type}_{seed}.pkl') and 
            os.path.exists(f'{self.path_models}/model_{regressor_type}_{seed}.pkl')):
            # Load saved predictions
            y_pred_train_df = pd.read_pickle(f'{self.path_output}/y_pred_train_{regressor_type}_{seed}.pkl')
            y_pred_train = y_pred_train_df['composite_score_pred'].values
            
            # Load saved model and predict test
            print(f"Loading model for {regressor_type} with seed {seed}...")
            model = pickle.load(open(f'{self.path_models}/model_{regressor_type}_{seed}.pkl', 'rb'))
            y_pred_test = model.predict(X_test.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'))
            
        else:
            # Ensure Y_train is a 1D array
            if isinstance(Y_train, pd.DataFrame):
                Y_train = Y_train.values.ravel()
            elif isinstance(Y_train, pd.Series):
                Y_train = Y_train.values

            kf = GroupKFold(n_splits=5)

            for train_idx, val_idx in kf.split(X_train, Y_train, X_train['combined_fold']):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = Y_train[train_idx], Y_train[val_idx]

                # Train model using best parameters
                model = Regressor(**study.best_params, random_state=seed)
                model.fit(X_tr.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'), y_tr)

                # Make predictions on the validation set
                y_pred_val = model.predict(X_val.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'))
                y_pred_val = np.where(y_pred_val > 1000, np.floor(y_pred_val), np.ceil(y_pred_val))

                # Store the predictions for the validation fold
                y_pred_train[val_idx] = y_pred_val

            # Train final model on entire training set
            model = Regressor(**study.best_params, random_state=seed)
            model.fit(X_train.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'), Y_train)
            
            # Save model
            pickle.dump(model, open(f'{self.path_models}/model_{regressor_type}_{seed}.pkl', 'wb'))
            
            # Predict test set
            y_pred_test = model.predict(X_test.drop(columns=['uid_num', 'combined_fold', 'feature_category'], axis=1, errors='ignore'))

        # Convert predictions to DataFrame
        y_pred_train_df = pd.DataFrame(y_pred_train, columns=['composite_score_pred'])
        y_pred_test_df = pd.DataFrame(y_pred_test, columns=['composite_score_pred'])

        y_pred_train_df['composite_score_pred'] = np.where(y_pred_train_df['composite_score_pred'] > 1000, 
                                                          np.floor(y_pred_train_df['composite_score_pred']), 
                                                          np.ceil(y_pred_train_df['composite_score_pred']))
        y_pred_test_df['composite_score_pred'] = np.where(y_pred_test_df['composite_score_pred'] > 1000,
                                                           np.floor(y_pred_test_df['composite_score_pred']),
                                                           np.ceil(y_pred_test_df['composite_score_pred']))

        # Add unique IDs
        y_pred_train_df['uid'] = X_train_uid
        y_pred_test_df['uid'] = X_test_uid

        if not self.load_results or not os.path.exists(f'{self.path_output}/y_pred_train_{regressor_type}_{seed}.pkl'):
            # Save training predictions
            y_pred_train_df.to_pickle(f'{self.path_output}/y_pred_train_{regressor_type}_{seed}.pkl')

        # Calculate RMSE only for training
        train_rmse = root_mean_squared_error(Y_train, y_pred_train)

        return y_pred_train_df, y_pred_test_df, train_rmse

    def run_multiple_seeds(self, X_train, Y_train, X_train_uid, X_test, X_test_uid, studies, seeds, regressor_types):
        """
        Run the train_and_evaluate_model function for multiple seeds and consolidate results.

        Parameters:
        - X_train: Training features.
        - Y_train: Training labels.
        - X_train_uid: Unique IDs for training data.
        - X_test: Test features.
        - X_test_uid: Unique IDs for test data.
        - studies: List of study objects for optimization results.
        - seeds: List of seeds to iterate through.
        - regressor_types: List of regressor types to iterate through.

        Returns:
        - results: Dictionary with keys as seeds and values as tuples of (X_train_new, X_test_new).
        """
        results = {}

        for seed in seeds:
            X_train_new = X_train.copy()
            X_test_new = X_test.copy()

            for study, regressor_type in zip(studies, regressor_types):
                # Generate column names dynamically
                train_col_name = f"composite_score_{regressor_type}_{studies.index(study)}_seed_{seed}"
                test_col_name = f"composite_score_{regressor_type}_{studies.index(study)}_seed_{seed}"

                # Train and evaluate model
                y_pred_train, y_pred_test, train_rmse = self.train_and_evaluate_model(
                    X_train, Y_train, X_train_uid, X_test, X_test_uid, study, regressor_type, seed
                )

                # Add predictions to the DataFrames
                X_train_new[train_col_name] = y_pred_train['composite_score_pred']
                X_test_new[test_col_name] = y_pred_test['composite_score_pred']

                print(f"Completed {regressor_type} with seed {seed}: Train RMSE = {train_rmse}")

            results[seed] = (X_train_new, X_test_new)

        return results

    def rmse_by_age_group(self, y_true, y_pred, age_group, X_train_post):
        """
        Calculate RMSE for a specific age group.

        Parameters:
        - y_true: True labels.
        - y_pred: Predicted labels.
        - age_group: Age group to calculate RMSE for.
        - X_train_post: Processed training data.

        Returns:
        - rmse: RMSE for the specified age group.
        """
        age_group_indices = X_train_post[X_train_post['category'] == age_group].index
        y_true_age_group = y_true.loc[age_group_indices]
        y_pred_age_group = y_pred[age_group_indices]
        rmse = np.sqrt(mean_squared_error(y_true_age_group, y_pred_age_group))
        return rmse

    def blend_predictions_by_age(self, X_train_post, X_test_post, X_test, Y_train, predictions_train, predictions_test):
        """
        Blends predictions using softmax weights by age group and computes RMSE for the test set.

        Parameters:
        - X_train_post: Processed training data.
        - X_test_post: Processed test data.
        - X_test: Original test data.
        - Y_train: Training labels.
        - predictions_train: Dictionary of training predictions.
        - predictions_test: Dictionary of test predictions.

        Returns:
        - y_pred_test_blended: Blended predictions for the test set.
        """
        # Ensure indices match for predictions
        for key in predictions_test:
            predictions_test[key].index = X_test_post.index

        age_groups = X_train_post['category'].unique()
        rmse_results = {}

        for pred_name, y_pred in predictions_train.items():
            rmse_by_age = {
                age_group: self.rmse_by_age_group(Y_train, y_pred['composite_score_pred'], age_group, X_train_post)
                for age_group in age_groups
            }
            rmse_results[pred_name] = rmse_by_age

        # Compute weights using softmax of negative RMSE
        weights_by_age_group = {age_group: {} for age_group in age_groups}
        for age_group in age_groups:
            rmse_for_models = np.array([rmse_results[model][age_group] for model in predictions_train.keys()])
            softmax_weights = softmax(-rmse_for_models)  # Negative RMSE because lower is better
            for i, model in enumerate(predictions_train.keys()):
                weights_by_age_group[age_group][model] = softmax_weights[i]

        # Blend predictions for the test set
        y_pred_test_blended = np.zeros(len(X_test), dtype=float)
        for age_group in age_groups:
            age_group_indices = X_test_post[X_test_post['category'] == age_group].index
            blended_predictions = np.zeros(len(age_group_indices), dtype=float)
            for model, weight in weights_by_age_group[age_group].items():
                y_pred_test_model = predictions_test[model].loc[age_group_indices, 'composite_score_pred']
                blended_predictions += weight * y_pred_test_model.values
            y_pred_test_blended[age_group_indices] = blended_predictions
            
        return y_pred_test_blended

    @staticmethod
    def categorize_multiple_columns(dataframe, columns, suffix='_q', bins=4, quantiles=None):
        """
        Categorize columns into quantile-based bins.

        Parameters:
        - dataframe: The DataFrame to modify.
        - columns: List of columns to categorize.
        - suffix: Suffix for the new column names.
        - bins: Number of bins for quantile categorization.
        - quantiles: Predefined quantiles for categorization based on train data (for test data).

        Returns:
        - dataframe: Modified DataFrame with new categorized columns.
        - quantiles_dict: Dictionary of quantiles for each column (if quantiles is None).
        """
        quantiles_dict = {}
        for column in columns:
            if quantiles is None:
                # Calculate quantiles and ensure uniqueness
                quantiles_dict[column] = np.unique(
                    dataframe[column].quantile(np.linspace(0, 1, bins + 1)).values)
            bin_edges = quantiles[column] if quantiles else quantiles_dict[column]
            
            # Ensure bin edges are strictly monotonically increasing
            if len(bin_edges) <= 1 or not np.all(np.diff(bin_edges) > 0):
                raise ValueError(f"Non-monotonic bin edges for column: {column}. Adjust the data or number of bins.")
            
            dataframe[column + suffix] = pd.cut(dataframe[column], bins=bin_edges, labels=False, include_lowest=True)
        return dataframe, quantiles_dict

    def full_pipeline(self, results, X_train, Y_train, X_test):
        """
        Full pipeline to process data for each seed, categorize, blend, and compute final predictions.

        Parameters:
        - results: Dictionary of results from multiple seeds.
        - X_train: Training features.
        - Y_train: Training labels.
        - X_test: Test features.

        Returns:
        - final_blended_results_train: Blended training results.
        - final_blended_results_test: Blended test results.
        - final_predictions_train: Final training predictions.
        - final_predictions_test: Final test predictions.
        """
        final_blended_results_test = {}
        final_blended_results_train = {}
        final_predictions_train = {}
        final_predictions_test = {}
        
        for seed, (X_train_new, X_test_new) in results.items():

            if (self.load_results and 
                os.path.exists(f'{self.path_models}/lasso_model_{seed}.pkl') and
                os.path.exists(f'{self.path_models}/lasso_study_{seed}.pkl') and
                os.path.exists(f'{self.path_output}/y_pred_train_linear1_{seed}.pkl')):
                
                # Load saved study, model and predictions
                print(f"Loading Lasso model and predictions for seed {seed}...")
                study = pickle.load(open(f'{self.path_models}/lasso_study_{seed}.pkl', 'rb'))
                final_model = pickle.load(open(f'{self.path_models}/lasso_model_{seed}.pkl', 'rb'))
                y_pred_train_linear1 = pd.read_pickle(f'{self.path_output}/y_pred_train_linear1_{seed}.pkl')
                
                # Only predict test
                best_params = study.best_params
                best_quantile = best_params['quantile']
                final_train_quantile = X_train_new.astype(float).quantile(best_quantile)
                X_test_lin = X_test_new.fillna(final_train_quantile)
                
                y_pred_test_linear1 = final_model.predict(X_test_lin.drop(columns=['uid_num', 'combined_fold', 'feature_category'], errors='ignore'))
                y_pred_test_linear1 = pd.DataFrame(y_pred_test_linear1, columns=['composite_score_linear1'])
                
            else:
                # LASSO - predict train
                def lasso_objective(trial):
                    alpha = trial.suggest_loguniform('alpha', 0.1, 0.5)
                    quantile = trial.suggest_float('quantile', 0.005, 0.5)

                    X_train_lin = X_train_new.astype(float)
                    train_quantile = X_train_lin.quantile(quantile)
                    X_train_lin = X_train_lin.fillna(train_quantile)

                    X_test_lin = X_test_new.astype(float)
                    X_test_lin = X_test_lin.fillna(train_quantile)

                    kf = GroupKFold(n_splits=5)
                    fold_errors = []

                    for train_idx, val_idx in kf.split(X_train_lin, Y_train, groups=X_train_lin['combined_fold']):
                        X_tr, X_val = X_train_lin.iloc[train_idx], X_train_lin.iloc[val_idx]
                        y_tr, y_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

                        X_tr = X_tr.drop(columns=['combined_fold', 'feature_category'], errors='ignore')
                        X_val = X_val.drop(columns=['combined_fold', 'feature_category'], errors='ignore')

                        linear_reg = Lasso(alpha=alpha, random_state=5, max_iter=100000, tol=1e-2)
                        linear_reg.fit(X_tr, y_tr)

                        y_pred_val = linear_reg.predict(X_val)
                        fold_error = np.sqrt(mean_squared_error(y_val, y_pred_val))
                        fold_errors.append(fold_error)

                    return np.mean(fold_errors)

                # Create and optimize study
                study = optuna.create_study(direction='minimize',
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5),
                                        sampler=optuna.samplers.TPESampler(seed=1))
                study.optimize(lasso_objective, n_trials=self.n_trials)

                best_params = study.best_params
                best_alpha = best_params['alpha']
                best_quantile = best_params['quantile']

                final_train_quantile = X_train_new.astype(float).quantile(best_quantile)
                X_train_lin = X_train_new.fillna(final_train_quantile)
                X_test_lin = X_test_new.fillna(final_train_quantile)

                final_model = Lasso(alpha=best_alpha, random_state=5, max_iter=100000, tol=1e-2)
                final_model.fit(X_train_lin.drop(columns=['uid_num', 'combined_fold', 'feature_category'], errors='ignore'), Y_train)

                y_pred_train_linear1 = final_model.predict(X_train_lin.drop(columns=['uid_num', 'combined_fold', 'feature_category'], errors='ignore'))
                y_pred_train_linear1 = pd.DataFrame(y_pred_train_linear1, columns=['composite_score_linear1'])

                y_pred_test_linear1 = final_model.predict(X_test_lin.drop(columns=['uid_num', 'combined_fold', 'feature_category'], errors='ignore'))
                y_pred_test_linear1 = pd.DataFrame(y_pred_test_linear1, columns=['composite_score_linear1'])

                # Save model, study and predictions
                pickle.dump(final_model, open(f'{self.path_models}/lasso_model_{seed}.pkl', 'wb'))
                pickle.dump(study, open(f'{self.path_models}/lasso_study_{seed}.pkl', 'wb'))
                y_pred_train_linear1.to_pickle(f'{self.path_output}/y_pred_train_linear1_{seed}.pkl')

            # rounding
            y_pred_train_linear1['composite_score_linear1'] = np.where(y_pred_train_linear1['composite_score_linear1'] > 1000,
                                                                    np.floor(y_pred_train_linear1['composite_score_linear1']),
                                                                    np.ceil(y_pred_train_linear1['composite_score_linear1']))
            y_pred_test_linear1['composite_score_linear1'] = np.where(y_pred_test_linear1['composite_score_linear1'] > 1000,
                                                            np.floor(y_pred_test_linear1['composite_score_linear1']),
                                                            np.ceil(y_pred_test_linear1['composite_score_linear1']))
            
            # added lasso predictions to X_train and X_test
            X_train_post = pd.concat([X_train_new, y_pred_train_linear1], axis=1)
            X_test_post = pd.concat([X_test_new, y_pred_test_linear1], axis=1)

            # prediction columns
            prediction_columns = [col for col in X_train_post.columns if col.startswith("composite_score_") and not col.endswith("_diff")]
            
            # Categorize X_train_new and determine quantiles
            X_train_post, quantiles_dict = self.categorize_multiple_columns(X_train_post, prediction_columns)

            # Categorize X_test_new using X_train_new quantiles
            X_test_post, _ = self.categorize_multiple_columns(X_test_post, prediction_columns, quantiles=quantiles_dict)
            
            # Create combined category column
            X_train_post['category'] = X_train_post['narrowed_age_12'].astype(str)
            X_test_post['category'] = X_test_post['narrowed_age_12'].astype(str)

            # Retain top categories
            X_train_post['category_top20'] = X_train_post['category'].apply(
                lambda x: x if x in X_train_post['category'].value_counts().head(6).index else 'other'
            )
            X_train_post['category'] = X_train_post['category_top20']
            X_test_post['category_top20'] = X_test_post['category'].apply(
                lambda x: x if x in X_train_post['category'].value_counts().head(6).index else 'other'
            )
            X_test_post['category'] = X_test_post['category_top20']

            # Predictions dictionary
            predictions_train = {
                col: pd.DataFrame(X_train_post[col].values, index=X_train_post.index, columns=['composite_score_pred'])
                for col in prediction_columns + ['composite_score_linear1']
            }
            predictions_test = {
                col: pd.DataFrame(X_test_post[col].values, index=X_test_post.index, columns=['composite_score_pred'])
                for col in prediction_columns + ['composite_score_linear1']
            }

            # Blend predictions by age group
            y_pred_test_blended = self.blend_predictions_by_age(
                X_train_post, 
                X_test_post, 
                X_test, 
                Y_train, 
                predictions_train, 
                predictions_test
            )

            y_pred_train_blended = self.blend_predictions_by_age(
                X_train_post, 
                X_train_post, 
                X_train, 
                Y_train, 
                predictions_train, 
                predictions_train
            )

            final_blended_results_train[seed] = y_pred_train_blended
            final_blended_results_test[seed] = y_pred_test_blended
            final_predictions_train[seed] = predictions_train
            final_predictions_test[seed] = predictions_test

        return final_blended_results_train, final_blended_results_test, final_predictions_train, final_predictions_test

    def process(self):
        """
        Main process to load data, optimize models, and generate predictions.

        Returns:
        - prediction_results: Dictionary containing blended test and train results, model predictions, and additional info.
        """
        # Load data
        X_train = pd.read_pickle(f'{self.path_data_processed}X_train.pkl')
        Y_train = pd.read_pickle(f'{self.path_data_processed}Y_train.pkl')
        X_train_uid = pd.read_pickle(f'{self.path_data_processed}X_train_uid.pkl')
        X_test_uid = pd.read_pickle(f'{self.path_data_processed}X_test_uid.pkl')
        X_test = pd.read_pickle(f'{self.path_data_processed}X_test.pkl')

        if self.load_results and os.path.exists(f'{self.path_models}/studies.pkl'):

            # Load existing studies
            print("Loading existing studies...")
            with open(f'{self.path_models}/studies.pkl', 'rb') as f:
                studies_dict = pickle.load(f)
                study_lgbm_gbdt = studies_dict['lgbm_gbdt']
                study_cat_rmse = studies_dict['cat_rmse']
                study_x = studies_dict['xgboost']
                
        else:

            # 'gbdt' boosting type
            study_lgbm_gbdt = optuna.create_study(direction="minimize",
                                                pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5),
                                                sampler=optuna.samplers.TPESampler(seed=1))
            study_lgbm_gbdt.optimize(self.objective_lgbm(boosting_type='gbdt', X_train=X_train.drop(columns=['feature_category'], errors='ignore'), Y_train=Y_train), n_trials=self.n_trials, n_jobs=1)

            # RMSE
            selected_metric = 'rmse'
            objective = self.objective_catboost(metric=selected_metric, X_train=X_train.drop(columns=['feature_category'], errors='ignore'), Y_train=Y_train)

            study_cat_rmse = optuna.create_study(
                direction="minimize" if selected_metric != 'map' else "maximize",
                                                pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5),
                                                sampler=optuna.samplers.TPESampler(seed=1)
            )
            study_cat_rmse.optimize(objective, n_trials=self.n_trials, n_jobs=1)
                
            study_x = optuna.create_study(direction="minimize",
                                                    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_startup_trials=5),
                                                sampler=optuna.samplers.TPESampler(seed=1))
            study_x.optimize(lambda trial: self.objective_xgboost(trial, X_train.drop(columns=['feature_category'], errors='ignore'),
                                                                  Y_train), n_trials=self.n_trials, n_jobs=1)

            # Save studies
            studies_dict = {
                'lgbm_gbdt': study_lgbm_gbdt,
                'cat_rmse': study_cat_rmse,
                'xgboost': study_x
            }
            with open(f'{self.path_models}/studies.pkl', 'wb') as f:
                pickle.dump(studies_dict, f)

        # Run nested models
        seeds = list(range(self.n_seeds))
        studies = [study_cat_rmse, study_lgbm_gbdt, study_x]
        regressor_types = ['catboost', 'lgbm', 'xgboost']
        results = self.run_multiple_seeds(
            X_train=X_train.drop(columns=['feature_category'], errors='ignore'),
            Y_train=Y_train,
            X_train_uid=X_train_uid,
            X_test=X_test.drop(columns=['feature_category'], errors='ignore'),
            X_test_uid=X_test_uid,
            studies=studies,
            seeds=seeds,
            regressor_types=regressor_types
        )

        # Run the full pipeline
        blended_results_train, blended_results_test, predictions_train, predictions_test = self.full_pipeline(
            results, 
            X_train, 
            Y_train,
            X_test
        )

        @staticmethod
        def compute_composite_score(blended_results, Y_train, n_seeds):
            """
            Compute the composite score by averaging and applying constraints.

            :return: Updated blended_results DataFrame with composite score column.
            """
            # Take the average across seeds
            blended_results['composite_score'] = blended_results.iloc[:, :n_seeds].mean(axis=1)

            # Truncate values below the minimum score from Y_train
            min_score = Y_train['composite_score'].min()
            cols = blended_results.columns[:n_seeds]
            blended_results[cols] = blended_results[cols].applymap(lambda x: max(x, min_score))

            # Round up composite scores
            blended_results['composite_score'] = np.ceil(blended_results['composite_score'])

            return blended_results

        blended_results_test = compute_composite_score(
            blended_results=pd.DataFrame(blended_results_test),
            Y_train=Y_train,
            n_seeds=self.n_seeds
        )
        
        blended_results_train = compute_composite_score(
            blended_results=pd.DataFrame(blended_results_train),
            Y_train=Y_train,
            n_seeds=self.n_seeds
        )
        
        # Add year and uid to the results
        test_info = pd.DataFrame(X_test_uid.copy())
        test_info['year'] = X_test['year']
        train_info = pd.DataFrame(X_train_uid.copy())
        train_info['year'] = X_train['year']
        train_info['combined_fold'] = X_train['combined_fold']

        print(blended_results_test.head())

        # Create dict with results
        prediction_results = {
            'blended_test': blended_results_test,
            'blended_train': blended_results_train,
            'models_train': predictions_train,
            'models_test': predictions_test,
            'test_info': test_info,
            'train_info': train_info
        }

        return prediction_results
