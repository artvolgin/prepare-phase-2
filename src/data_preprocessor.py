import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import warnings
import gc

class DataPreprocessor:
    def __init__(self,
                 path_data_raw='../data/raw/',
                 path_data_processed='../data/processed/', 
                 path_output='../output/',
                 remove_na_obs=False,
                 drop_random_features=False,
                 replace_outliers=False,
                 path_inference_dataset=None):
        """
        Initializes the DataPreprocessor with specified data directories and configurations.
        
        Parameters:
        - path_data_raw (str): Directory where raw data CSV files are located.
        - path_data_processed (str): Directory where processed data will be saved.
        - remove_many_NAs (bool): Whether to remove observations with many NAs.
        - drop_random_features (bool): Whether to randomly drop features.
        - use_inference (bool): Whether to use inference data.
        """
        # Suppress warnings
        warnings.filterwarnings("ignore")
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        # Enable garbage collection
        gc.enable()
        
        # Define age categories and their ranges
        self.age_category_ranges = {
            '0. 49 or younger': (40, 49),
            '1. 50–59': (50, 59),
            '2. 60–69': (60, 69),
            '3. 70–79': (70, 79),
            '4. 80+': (80, 100),  # Assuming 100 as the upper bound for simplicity
        }
        
        # Define ordinal columns and their order
        self.ordinal_columns = {
            'glob_hlth_03': ['1. Excellent', '2. Very good', '3. Good', '4. Fair', '5. Poor'],
            'glob_hlth_12': ['1. Excellent', '2. Very good', '3. Good', '4. Fair', '5. Poor'],
            'bmi_03': ['1. Underweight', '2. Normal weight', '3. Overweight', '4. Obese', '5. Morbidly obese'],
            'bmi_12': ['1. Underweight', '2. Normal weight', '3. Overweight', '4. Obese', '5. Morbidly obese'],
            'memory_12': ['1. Excellent', '2. Very good', '3. Good', '4. Fair', '5. Poor'],
            'rrelgimp_03': ['1.very important', '2.somewhat important', '3.not important'],
            'rrelgimp_12': ['1.very important', '2.somewhat important', '3.not important'],
            'satis_ideal_12': ['1. Agrees', '2. Neither agrees nor disagrees', '3. Disagrees'],
            'satis_excel_12': ['1. Agrees', '2. Neither agrees nor disagrees', '3. Disagrees'],
            'satis_fine_12': ['1. Agrees', '2. Neither agrees nor disagrees', '3. Disagrees'],
            'cosas_imp_12': ['1. Agrees', '2. Neither agrees nor disagrees', '3. Disagrees'],
            'wouldnt_change_12': ['1. Agrees', '2. Neither agrees nor disagrees', '3. Disagrees'],
            'decis_personal_12': ['1. A lot', '2. A little', '3. None'], 
            'rrfcntx_m_12': ['1.Almost every day', '2.4 or more times a week', '3.2 or 3 times a week', '4.Once a week', '5.2 or 3 times a month', '6.Once a month', '7.Almost Never, sporadic', '8. Never'],
            'rsocact_m_12': ['1.Almost every day', '2.4 or more times a week', '3.2 or 3 times a week', '4.Once a week', '5.2 or 3 times a month', '6.Once a month', '7.Almost Never, sporadic', '8. Never']
        }
        
        # Education categories
        self.edu_categories = {
            '0. No education': 0,
            '1. 1–5 years': 2,
            '2. 6 years': 7,
            '3. 7–9 years': 8,
            '4. 10+ years': 10
        }
        
        # Define categorical columns for one-hot encoding
        self.categorical_columns = [
            'married_03', 'married_12', 'urban_03',
            'urban_12', 'employment_03', 'employment_12',
            'ragender', 'sgender_03', 'sgender_12',
            'rameduc_m', 'rafeduc_m', 'rjlocc_m_03',
            'rjlocc_m_12', 'rrelgwk_12', 'a22_12', 
            'a33b_12', 'a34_12', 'j11_12',
            'rjobend_reason_03', 'rjobend_reason_12',
            'age_03', 'age_12', 'decis_famil_03',
            'decis_famil_12'   
        ]
        
        # Living children categories
        self.living_children_categories = {
            '0. No children': 0,
            '1. 1 or 2': 1.5,
            '2. 3 or 4': 3.5,
            '3. 5 or 6': 5.5,
            '4. 7+': 7
        }

        # Variables for outlier replacement
        self.income_cols = [
            'hincome_03', 'hinc_business_03', 'hinc_rent_03', 'hinc_assets_03', 'hinc_cap_03',
            'hincome_12', 'hinc_business_12', 'hinc_rent_12', 'hinc_assets_12', 'hinc_cap_12'
        ]
        
        self.essential_columns = ['uid', 'age_03', 'age_12']
        
        # Directories for data
        self.path_data_raw = path_data_raw
        self.path_data_processed = path_data_processed
        self.path_output = path_output
        
        # Initialize data attributes
        self.test_features = None
        self.train_features = None
        self.train_labels = None
        self.features_data = None
        self.submission_format = None
        
        self.remove_na_obs = remove_na_obs
        self.drop_random_features = drop_random_features
        self.replace_outliers = replace_outliers

        self.path_inference_dataset = path_inference_dataset
        
    def generate_year_mapping(self, data_features_participation, long, labels):
        """
        Generate a mapping from unique year combinations to integers.
        
        Parameters:
        - data_features_participation (pd.DataFrame): DataFrame containing participation data.
        - long (pd.DataFrame): Long-format DataFrame with 'uid' and 'year'.
        - labels (pd.DataFrame): Labels DataFrame containing 'uid' and 'year'.
        
        Returns:
        - dict: A dictionary mapping year combinations to unique integers.
        """
        # Merge data_features_participation with selected columns from long
        data_features_participation_ = data_features_participation.merge(
            long[['uid', 'year']], on='uid', how='left'
        )
        
        # Concatenate and aggregate to get unique combinations of years
        participation = pd.concat([
            data_features_participation_[['uid', 'year']], labels[['uid', 'year']]
        ], axis=0)
        
        participation = participation.groupby(['uid']).agg(
            year=('year', lambda x: list(x))
        ).reset_index()
    
        # Extract unique combinations of years as strings
        unique_years = participation['year'].apply(lambda x: str(sorted(x))).unique()
    
        # Create year mapping automatically
        year_mapping = {year_combination: i + 1 for i, year_combination in enumerate(sorted(unique_years))}
        return year_mapping
    
    def prepare_participation(self, data_features_participation, long, labels, year_mapping):
        """
        Prepare participation data by merging and mapping year combinations.
        
        Parameters:
        - data_features_participation (pd.DataFrame): DataFrame containing participation data.
        - long (pd.DataFrame): Long-format DataFrame with 'uid' and 'year'.
        - labels (pd.DataFrame): Labels DataFrame containing 'uid' and 'year'.
        - year_mapping (dict): Mapping from year combinations to integers.
        
        Returns:
        - pd.DataFrame: DataFrame with 'uid' and 'year_participation'.
        """
        # Merge data_features_participation with selected columns from long
        data_features_participation_ = data_features_participation.merge(
            long[['uid', 'year']], on='uid', how='left'
        )
        
        # Concatenate and aggregate to get participation counts and years
        participation = pd.concat([
            data_features_participation_[['uid', 'year']], labels[['uid', 'year']]
        ], axis=0)
        
        participation = participation.groupby(['uid']).agg(
            participation=('year', 'count'),
            year=('year', lambda x: list(x))
        ).reset_index()
    
        # Recode year combinations to numbers
        participation['year'] = participation['year'].apply(lambda x: year_mapping.get(str(sorted(x)), np.nan))
        
        # Rename year column to year_participation
        participation.rename(columns={'year': 'year_participation'}, inplace=True)
    
        # Keep only year_participation column
        participation = participation[['uid', 'year_participation']]
    
        return participation
    
    def narrow_age(self, row):
        """
        Narrow down age if the person remained in the same category.
        
        Parameters:
        - row (pd.Series): A row from the DataFrame.
        
        Returns:
        - float or None: The narrowed age or None if age data is missing.
        """
        age_2003 = row['age_03']
        age_2012 = row['age_12']
        
        # Handle missing values
        if pd.isna(age_2003) or pd.isna(age_2012):
            return None  # Can't determine age if one or both are missing
        
        # Check if the category didn't change
        if age_2003 == age_2012:
            # If the category didn't change, assume they're at the beginning of the range
            return self.age_category_ranges.get(age_2003, (np.nan, np.nan))[0]
        else:
            # If the category changed, narrow down the age to the end of the initial range (2003)
            # and the start of the new range (2012)
            age_2003_range = self.age_category_ranges.get(age_2003, (np.nan, np.nan))[1]  # End of the 2003 range
            age_2012_range = self.age_category_ranges.get(age_2012, (np.nan, np.nan))[0]  # Start of the 2012 range
            if np.isnan(age_2003_range) or np.isnan(age_2012_range):
                return None
            return np.mean([age_2003_range, age_2012_range])
        
    def load_data(self):
        """
        Load all necessary CSV files from the specified data directory.
        """

        self.test_features = pd.read_csv(self.path_inference_dataset)
        self.train_features = pd.read_csv(f'{self.path_data_raw}train_features.csv')
        self.train_labels = pd.read_csv(f'{self.path_data_raw}train_labels.csv')
        self.features_data = pd.concat([self.train_features, self.test_features], axis=0).reset_index(drop=True)    
        self.submission_format = pd.read_csv(f'{self.path_data_raw}submission_format.csv')
    
    def convert_ordinal_columns(self):
        """
        Convert ordinal columns to ordered numeric codes, checking if they exist first.
        """
        for col, order in self.ordinal_columns.items():
            if col in self.features_data.columns:
                cat_type = CategoricalDtype(categories=order, ordered=True)
                self.features_data[col] = self.features_data[col].astype(cat_type)
                self.features_data[col] = self.features_data[col].cat.codes.replace(-1, pd.NA) + 1  
                # Convert to integers
                self.features_data[col] = self.features_data[col].astype(pd.Int64Dtype())
            else:
                print(f"Column '{col}' not found in the dataset.")
    
    def recode_columns(self):
        """
        Recode education and living children columns to numerical values.
        """
        # Recode edu_gru_03 and edu_gru_12
        if 'edu_gru_03' in self.features_data.columns:
            self.features_data['edu_gru_03'] = self.features_data['edu_gru_03'].replace(self.edu_categories)
        if 'edu_gru_12' in self.features_data.columns:
            self.features_data['edu_gru_12'] = self.features_data['edu_gru_12'].replace(self.edu_categories)
        
        # Recode n_living_child_03 and n_living_child_12
        if 'n_living_child_03' in self.features_data.columns:
            self.features_data['n_living_child_03'] = self.features_data['n_living_child_03'].replace(self.living_children_categories)
        if 'n_living_child_12' in self.features_data.columns:
            self.features_data['n_living_child_12'] = self.features_data['n_living_child_12'].replace(self.living_children_categories)
    
    def one_hot_encode_categorical_columns(self):
        """
        Perform one-hot encoding on categorical columns, preserving NaNs.
        """
        # Check if categorical columns exist before applying one-hot encoding
        existing_categorical_columns = [col for col in self.categorical_columns if col in self.features_data.columns]
        
        original_columns = set(self.features_data.columns)
        if existing_categorical_columns:
            # Convert existing categorical columns to dummy variables (one-hot encoding), preserving NaNs
            self.features_data = pd.get_dummies(
                self.features_data,
                columns=existing_categorical_columns,
                drop_first=False,
                dummy_na=True  # Add a dummy column for NaN values
            )
            # new_columns = set(self.features_data.columns) - original_columns
        else:
            print("No categorical columns found for one-hot encoding.")
            
        # Convert all dummy variables to integers, keeping NaNs represented appropriately
        dummy_columns = self.features_data.select_dtypes(include=['bool', 'uint8']).columns
        self.features_data[dummy_columns] = self.features_data[dummy_columns].astype(pd.Int64Dtype())  # Use nullable integer type to allow NaNs
        
        # Rename columns to remove the trailing "_0" and ".0" from one-hot encoding    
        self.features_data.columns = self.features_data.columns.str.replace(r"_0$", "", regex=True)
        self.features_data.columns = self.features_data.columns.str.replace(r".0$", "", regex=True)
    
    def compute_differences(self):
        """
        Compute the difference between corresponding columns from different years.
        """
        # Identify columns except 'uid'
        all_columns = [col for col in self.features_data.columns if col != 'uid']
        
        # Select the columns that are available in both years
        available_columns = [column for column in all_columns if column.endswith('_12') and 
                             (column.replace('_12', '') + '_03') in self.features_data.columns and 
                             column not in ['narrowed_age_12', 'narrowed_age_03']]
                
        # Compute the difference between the columns that are available in both years
        for column in available_columns:
            base_column = column.replace('_12', '')
            if (base_column + '_03') in self.features_data.columns and column in self.features_data.columns:
                self.features_data[base_column + '_diff'] = self.features_data[base_column + '_03'] - self.features_data[column]
    
    def transform_wide_to_long(self, df, is_train=True):
        """
        Transform wide format data to long format.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to transform.
        - is_train (bool): Flag indicating whether the data is for training.
        
        Returns:
        - pd.DataFrame: Long-format DataFrame with an additional 'year' column.
        """
        df_03 = df.filter(regex='(uid|_03$)', axis=1).copy()
        df_12 = df.filter(regex='(uid|_12$)', axis=1).copy()
        
        # Assign year
        df_03['year'] = 2003
        df_12['year'] = 2012
        
        # Remove suffixes from column names
        df_03.columns = df_03.columns.str.replace('_03$', '', regex=True)
        df_12.columns = df_12.columns.str.replace('_12$', '', regex=True)
        
        # Remove rows missing in all columns except uid and year
        cols_excluding_uid_year = df_03.columns.drop(['uid', 'year'])
        df_03 = df_03.dropna(subset=cols_excluding_uid_year.tolist(), how='all')
        cols_excluding_uid_year = df_12.columns.drop(['uid', 'year'])
        df_12 = df_12.dropna(subset=cols_excluding_uid_year.tolist(), how='all')
        
        # Concatenate the two years
        df_long = pd.concat([df_03, df_12], axis=0)
        
        return df_long
    
    def generate_and_prepare_participation(self, train_long, test_long):
        """
        Generate year mapping and prepare participation for training and testing data.
        
        Parameters:
        - train_long (pd.DataFrame): Long-format training DataFrame.
        - test_long (pd.DataFrame): Long-format testing DataFrame.
        """
        # Group by 'uid' and count participation
        train_participation_counts = train_long.groupby('uid').size().reset_index(name='participation')
        test_participation_counts = test_long.groupby('uid').size().reset_index(name='participation')
        
        # Filter participation ==1 and ==2 and merge with years (if needed)
        train_participation_1 = train_participation_counts[train_participation_counts['participation'] == 1]
        train_participation_1 = train_participation_1.merge(train_long[['uid', 'year']], on='uid', how='left')
        
        train_participation_2 = train_participation_counts[train_participation_counts['participation'] == 2]
        train_participation_2 = train_participation_2.merge(train_long[['uid', 'year']], on='uid', how='left')
        
        test_participation_1 = test_participation_counts[test_participation_counts['participation'] == 1]
        test_participation_1 = test_participation_1.merge(test_long[['uid', 'year']], on='uid', how='left')
        
        test_participation_2 = test_participation_counts[test_participation_counts['participation'] == 2]
        test_participation_2 = test_participation_2.merge(test_long[['uid', 'year']], on='uid', how='left')
        
        # Generate year mapping based on training data
        year_mapping_train = self.generate_year_mapping(
            data_features_participation=train_participation_counts,
            long=train_long,
            labels=self.train_labels
        )
        
        # Training data preparation
        train_participation = self.prepare_participation(
            data_features_participation=train_participation_counts,
            long=train_long, 
            labels=self.train_labels,
            year_mapping=year_mapping_train
        )
        
        # Test data preparation (use year mapping from training phase)
        test_participation = self.prepare_participation(
            data_features_participation=test_participation_counts,
            long=test_long, 
            labels=self.submission_format,
            year_mapping=year_mapping_train
        )
        
        # Concatenate train and test participation
        participation = pd.concat([train_participation, test_participation], axis=0)
        
        # Merge with features_data
        self.features_data = self.features_data.merge(participation[['uid', 'year_participation']], on=['uid'], how='left')
        
        # One-hot encode 'year_participation'
        self.features_data = pd.get_dummies(self.features_data, columns=['year_participation'], drop_first=False)
        part_cols = [col for col in self.features_data.columns if 'year_participation' in col]
        self.features_data[part_cols] = self.features_data[part_cols].astype(int)
        
        # Merge again 'year_participation' if needed (as in original code)
        self.features_data = self.features_data.merge(participation[['uid', 'year_participation']], on=['uid'], how='left')
    
    def process(self):
        """
        Execute the entire data processing pipeline.
        """
        # Load data
        self.load_data()
        
        # Optionally remove observations with many NAs
        if self.remove_na_obs:
            print("... Removing observations with many NAs ...")
            # Filter columns for 2003 and 2012
            features_data_03 = self.features_data.loc[:, self.features_data.columns.str.endswith('_03')]
            features_data_12 = self.features_data.loc[:, self.features_data.columns.str.endswith('_12')]

            # Calculate NA counts and determine which rows to keep for 2003
            features_data_03['n_NA_03'] = features_data_03.isna().sum(axis=1)
            features_data_03['NA_keep_03'] = (
                (features_data_03['n_NA_03'] < int(features_data_03.shape[1] / 2)) |
                (features_data_03['n_NA_03'] == 75)
            )

            # Calculate NA counts and determine which rows to keep for 2012
            features_data_12['n_NA_12'] = features_data_12.isna().sum(axis=1)
            features_data_12['NA_keep_12'] = (
                (features_data_12['n_NA_12'] < int(features_data_12.shape[1] / 2)) |
                (features_data_12['n_NA_12'] > 100)
            )

            # Combine keep flags and filter the data
            self.features_data['NA_keep'] = features_data_03['NA_keep_03'] & features_data_12['NA_keep_12']
            self.features_data = self.features_data[self.features_data['NA_keep'] == True].copy()
            self.features_data.drop(columns=['NA_keep'], inplace=True, errors='ignore')

        # Optionally replace outliers in income variables at 99th quantile
        if self.replace_outliers:
            print("... Replacing outliers in income variables at 99th quantile ...")
            for col in self.income_cols:
                if col in self.features_data.columns:
                    q99 = self.features_data[col].dropna().quantile(0.99)
                    self.features_data[col] = np.where(self.features_data[col] > q99, q99, self.features_data[col])

        # Optionally drop random features
        if self.drop_random_features:
            self.drop_random_features_method(drop_fraction=0.5)

        # Apply narrow_age function
        self.features_data['narrowed_age_03'] = self.features_data.apply(self.narrow_age, axis=1)
        self.features_data['narrowed_age_12'] = self.features_data['narrowed_age_03'] + 9
        
        # Convert ordinal columns
        self.convert_ordinal_columns()
        
        # Recode education and living children columns
        self.recode_columns()
        
        # One-hot encode categorical columns
        self.one_hot_encode_categorical_columns()
        
        # Compute differences between years
        self.compute_differences()
        
        # Transform wide to long format for training and testing data
        train_long = self.transform_wide_to_long(self.train_features, is_train=True)
        test_long = self.transform_wide_to_long(self.test_features, is_train=False)
        
        # Generate and prepare participation mappings
        self.generate_and_prepare_participation(train_long, test_long)
        
        # Save the processed features_data
        self.save_features()
    
    def save_features(self):
        """
        Save the processed features_data to a pickle file in the processed directory.
        """
        self.features_data.to_pickle(f'{self.path_data_processed}features_data.pkl')
        print(f"Processed data saved to {self.path_data_processed}features_data.pkl")

    def drop_random_features_method(self, drop_fraction=0.2, random_state=42):
        """
        Randomly drops a fraction of features while preserving essential columns.
        
        Parameters:
        - drop_fraction (float): Fraction of non-essential features to drop (0.0 to 1.0)
        - random_state (int): Random seed for reproducibility
        """
        # Get all columns except essential ones
        droppable_columns = [col for col in self.features_data.columns 
                           if col not in self.essential_columns]
        
        # Calculate number of columns to drop
        n_to_drop = int(len(droppable_columns) * drop_fraction)
        
        # Randomly select columns to drop
        np.random.seed(random_state)
        columns_to_drop = np.random.choice(droppable_columns, 
                                         size=n_to_drop, 
                                         replace=False)
        
        # Drop the selected columns
        self.features_data.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped {len(columns_to_drop)} features randomly")
