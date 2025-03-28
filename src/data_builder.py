# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import  KFold
from scipy.stats import entropy, hmean, gmean
from scipy.signal import find_peaks

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class DataBuilder:
    """
    A class used to build and process data for machine learning models.

    Attributes
    ----------
    include_advanced_stats : bool
        Flag to include advanced statistical features.
    include_pred_diff : bool
        Flag to include prediction differences.
    include_tabpfn : bool
        Flag to include tabpfn results.
    path_data_processed : str
        Path to the processed data directory.
    path_data_raw : str
        Path to the raw data directory.
    path_inference_dataset : str or None
        Path to the inference dataset.
    path_output : str
        Path to the output directory.
    path_quantiles_tabpfn : str
        Path to the quantiles tabpfn file.
    """

    def __init__(self,
                 include_advanced_stats=True,
                 include_pred_diff=True,
                 include_tabpfn=True,
                 path_data_processed='../data/processed/',
                 path_data_raw='../data/raw/',
                 path_inference_dataset=None,
                 path_output='../output/',
                 path_quantiles_tabpfn='../output/quantiles_tabpfn.pkl'):
        """
        Initializes the DataBuilder with specified parameters.

        Parameters
        ----------
        include_advanced_stats : bool, optional
            Whether to include advanced statistical features (default is True).
        include_pred_diff : bool, optional
            Whether to include prediction differences (default is True).
        include_tabpfn : bool, optional
            Whether to include tabpfn results (default is True).
        path_data_processed : str, optional
            Path to the processed data directory (default is '../data/processed/').
        path_data_raw : str, optional
            Path to the raw data directory (default is '../data/raw/').
        path_inference_dataset : str or None, optional
            Path to the inference dataset (default is None).
        path_output : str, optional
            Path to the output directory (default is '../output/').
        path_quantiles_tabpfn : str, optional
            Path to the quantiles tabpfn file (default is '../output/quantiles_tabpfn.pkl').
        """
        self.include_advanced_stats = include_advanced_stats
        self.include_pred_diff = include_pred_diff
        self.include_tabpfn = include_tabpfn
        self.path_data_processed = path_data_processed
        self.path_inference_dataset = path_inference_dataset
        self.path_data_raw = path_data_raw
        self.path_output = path_output
        self.path_quantiles_tabpfn = path_quantiles_tabpfn

    def advanced_stats(self, df, group_vars):
        """
        Calculate advanced statistical features for the given DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the data.
        group_vars : list
            List of columns to group by.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with advanced statistical features.
        """
        # Calculate standard aggregations separately
        grouped_stats = df.groupby(group_vars)['composite_score'].agg([
            'min', 'max', 'mean', 'std', 'skew', 'median'
        ]).reset_index()
        
        # Custom aggregations - apply separately
        grouped_stats['range'] = df.groupby(group_vars)['composite_score'].apply(lambda x: x.max() - x.min()).values
        grouped_stats['iqr'] = df.groupby(group_vars)['composite_score'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).values
        grouped_stats['mad'] = df.groupby(group_vars)['composite_score'].apply(lambda x: (x - x.mean()).abs().mean()).values  # Manual calculation of MAD
        grouped_stats['cv'] = df.groupby(group_vars)['composite_score'].apply(lambda x: x.std() / x.mean() if x.mean() != 0 else 0).values
        grouped_stats['gini'] = df.groupby(group_vars)['composite_score'].apply(lambda x: sum([abs(i - j) for i in x for j in x]) / (2 * len(x) * sum(x)) if sum(x) != 0 else 0).values
        grouped_stats['entropy'] = df.groupby(group_vars)['composite_score'].apply(lambda x: entropy(pd.Series(x).value_counts(normalize=True))).values
        grouped_stats['harmonic_mean'] = df.groupby(group_vars)['composite_score'].apply(lambda x: hmean(x) if all(x > 0) else 0).values
        grouped_stats['geometric_mean'] = df.groupby(group_vars)['composite_score'].apply(lambda x: gmean(x) if all(x > 0) else 0).values
        grouped_stats['peaks'] = df.groupby(group_vars)['composite_score'].apply(lambda x: len(find_peaks(x)[0])).values
        grouped_stats['troughs'] = df.groupby(group_vars)['composite_score'].apply(lambda x: len(find_peaks(-x)[0])).values

        # Calculate quintiles separately
        quintile_labels = ['q1_20', 'q2_40', 'q3_60', 'q4_80']
        quintiles_df = df.groupby(group_vars)['composite_score'].apply(
            lambda x: pd.Series(x.quantile([0.2, 0.4, 0.6, 0.8]).values)
        ).unstack()
        quintiles_df.columns = quintile_labels
        quintiles_df.reset_index(inplace=True)

        # Merge quintiles with grouped stats
        grouped_stats = grouped_stats.merge(quintiles_df, on=group_vars, how='left')
        
        return grouped_stats
    
    def process(self):
        """
        Process the data by loading, transforming, and saving it.

        This method performs several operations including loading data,
        preprocessing, feature engineering, and saving the processed data.
        """
        # Load data
        full_train = pd.read_pickle(f'{self.path_data_processed}full_train.pkl')
        full_test = pd.read_pickle(f'{self.path_data_processed}full_test.pkl')
        quantiles_tabpfn = pd.read_pickle(self.path_quantiles_tabpfn)
        y_pred_diff_train_fin = pd.read_pickle(f'{self.path_output}pred_diff_train.pkl')
        y_pred_diff_test_fin = pd.read_pickle(f'{self.path_output}pred_diff_test.pkl')
        train_labels = pd.read_csv(f'{self.path_data_raw}train_labels.csv')

        # basic pre-processing
        X_train = full_train.copy()
        X_test = full_test.copy()

        # force object column 'uid' to integers
        X_train['uid_num'] = X_train['uid'].astype('category').cat.codes
        X_test['uid_num'] = X_test['uid'].astype('category').cat.codes

        Y_train = train_labels.drop(columns=['uid', 'year']).reset_index(drop=True)

        # Clean feature names by replacing special characters with underscores
        X_train.columns = X_train.columns.str.replace(r'[^\w]', '_', regex=True)
        X_test.columns = X_test.columns.str.replace(r'[^\w]', '_', regex=True)

        X_test_uid = X_test['uid']
        X_test = X_test.drop(columns=['uid']).reset_index(drop=True)

        X_train_uid = X_train['uid'].reset_index(drop=True)
        X_train = X_train.reset_index(drop=True).drop(columns=['uid'], axis=1)
        Y_train = Y_train.reset_index(drop=True)

        # Store original column names before any transformations
        original_columns = X_train.columns.tolist()

        # Define columns to take log transformation
        columns_to_log = [
            'hincome_03', 'hincome_12', 'hinc_business_03', 'hinc_business_12',
            'hinc_rent_03', 'hinc_rent_12', 'hinc_assets_03', 'hinc_assets_12',
            'hinc_cap_03', 'hinc_cap_12', 'rinc_pension_03', 'rinc_pension_12',
            'sinc_pension_03', 'sinc_pension_12',
            'hincome_diff','hinc_business_diff',
            'hinc_rent_diff', 'hinc_assets_diff'
        ]

        # Filter to only include columns that exist in the dataset
        columns_to_log = [col for col in columns_to_log if col in original_columns]

        # Apply log transformation, handling zero or negative values by replacing them with NaN
        for col in columns_to_log:
            X_train[col] = np.where(X_train[col] > 0, np.log(X_train[col]), np.nan)
            X_test[col] = np.where(X_test[col] > 0, np.log(X_test[col]), np.nan)
            
        # Add interaction and polynomial features for age if columns exist
        if 'narrowed_age_12' in original_columns:
            X_train['age_squared'] = X_train['narrowed_age_12'] ** 2
            X_test['age_squared'] = X_test['narrowed_age_12'] ** 2
            
            if 'hincome_12' in original_columns:
                X_train['age_income_interaction'] = X_train['narrowed_age_12'] * X_train['hincome_12']
                X_test['age_income_interaction'] = X_test['narrowed_age_12'] * X_test['hincome_12']

        # Ensure both train and test have the same columns in the same order
        all_columns = X_train.columns.tolist()
        X_train = X_train[all_columns]
        X_test = X_test[all_columns]

        # Modify the stats generation section to be conditional
        if self.include_advanced_stats:
            # Prepare composite score with the grouping variables for easier grouping
            df = pd.concat([X_train, Y_train[['composite_score']]], axis=1)
            
            # Define potential grouping variables
            potential_group_vars = [
                ['edu_gru_12'],
                ['narrowed_age_12'],
                ['year_participation'],
                ['bmi_12'],
                ['glob_hlth_12'], 
                ['reads_12'],
                ['n_living_child_12'],
                ['j11_12_Wood__mosaic__or_other_covering_1'],
                ['memory_12'],
                ['n_depr_12'],
                ['games_12'], 
                ['rrfcntx_m_12'],
                ['rsocact_m_12']
            ]
            
            # Filter to only use variables that exist in the dataframe
            valid_group_vars = [vars for vars in potential_group_vars 
                              if all(var in df.columns for var in vars)]
            
            # Iterate over valid grouping variables
            for group_var in valid_group_vars:
                grouped_stats = self.advanced_stats(df, group_var)
                # Flatten the list for the suffix in column names
                suffix = '_'.join(group_var)
                X_train = X_train.merge(grouped_stats, on=group_var, how='left', suffixes=('', f'_{suffix}'))
                X_test = X_test.merge(grouped_stats, on=group_var, how='left', suffixes=('', f'_{suffix}'))

        # Always add fold1 from y_pred_diff_train_fin
        fold1_df = y_pred_diff_train_fin[['uid', 'fold1']].copy()
        X_train = pd.concat([full_train['uid'], X_train], axis=1).merge(fold1_df, on=['uid'], how='left')
        X_train = X_train.drop(columns=['uid']).reset_index(drop=True)

        # Make pred_diff addition conditional
        if self.include_pred_diff:
            # Add prediction differences
            y_pred_diff_train_mod = y_pred_diff_train_fin.drop(columns=['fold1']).copy()  # Remove fold1 as it's already added
            X_train = pd.concat([full_train['uid'], X_train], axis=1).merge(y_pred_diff_train_mod, on=['uid'], how='left')
            X_train = X_train.drop(columns=['uid']).reset_index(drop=True)

            y_pred_diff_test_mod = y_pred_diff_test_fin.copy()
            X_test = pd.concat([full_test['uid'], X_test], axis=1).merge(y_pred_diff_test_mod, on=['uid'], how='left')
            X_test = X_test.drop(columns=['uid']).reset_index(drop=True)

        # missing fold in X_train replace with 0
        X_train['fold1'] = X_train['fold1'].fillna(0)
        kf_all = KFold(n_splits=5, shuffle=True, random_state=42)
        X_train['fold2'] = 0

        for fold, (_, val_idx) in enumerate(kf_all.split(X_train), start=1):
            X_train.loc[val_idx, 'fold2'] = fold
            
        X_train['combined_fold'] = X_train['fold1'].astype(str) + '_' + X_train['fold2'].astype(str)

        # drop fold1 and fold2
        X_train = X_train.drop(columns=['fold1', 'fold2'], axis=1)

        # adding tabpfn_results conditionally
        if self.include_tabpfn:
            X_train = pd.concat([X_train, X_train_uid], axis=1).merge(quantiles_tabpfn, on=['uid', 'year'], how='left').drop(columns=['uid'], axis=1)
            X_test = pd.concat([X_test, X_test_uid], axis=1).merge(quantiles_tabpfn, on=['uid', 'year'], how='left').drop(columns=['uid'], axis=1)

        # Save to pickle
        X_train.to_pickle(f'{self.path_data_processed}X_train.pkl')
        X_test.to_pickle(f'{self.path_data_processed}X_test.pkl')
        Y_train.to_pickle(f'{self.path_data_processed}Y_train.pkl')
        X_train_uid.to_pickle(f'{self.path_data_processed}X_train_uid.pkl')
        X_test_uid.to_pickle(f'{self.path_data_processed}X_test_uid.pkl')

