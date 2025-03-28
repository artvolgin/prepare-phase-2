import pandas as pd

def create_feature_category_list(df):
    """
    Create a list of dictionaries containing feature categories and their availability in years 03 and 12.
    
    Args:
        df: pandas DataFrame containing the features
    
    Returns:
        List of dicts with structure:
        [{'feature_name': str, 'category': str, 'year_03': bool, 'year_12': bool, 
          'count_03': int, 'count_12': int}, ...]
    """
    # Drop uid
    df = df.drop(columns=['uid'])
    # Get all columns that end with _03 or _12
    time_cols = [col for col in df.columns if col.endswith('_03') or col.endswith('_12')]
    
    # Extract base feature names by removing the _03 and _12 suffixes
    time_based_features = set()
    for col in time_cols:
        base_name = col[:-3]
        time_based_features.add(base_name)
    
    # Find columns that don't have _03 or _12 suffix
    non_time_features = [col for col in df.columns if not col.endswith('_03') and not col.endswith('_12')]
    
    result = []
    
    # Process time-based features
    for base_feature in time_based_features:
        feat_03 = f"{base_feature}_03"
        feat_12 = f"{base_feature}_12"
        
        vals_03 = set(df[feat_03].dropna().unique()) if feat_03 in df.columns else set()
        vals_12 = set(df[feat_12].dropna().unique()) if feat_12 in df.columns else set()
        
        all_categories = vals_03.union(vals_12)
        
        for category in all_categories:
            count_03 = df[df[feat_03] == category].shape[0] if feat_03 in df.columns else 0
            count_12 = df[df[feat_12] == category].shape[0] if feat_12 in df.columns else 0
            
            entry = {
                'feature_name': base_feature,
                'category': str(category),
                'year_03': feat_03 in df.columns,
                'year_12': feat_12 in df.columns,
                'count_03': count_03,
                'count_12': count_12
            }
            result.append(entry)
    
    # Process non-time-based features
    for feature in non_time_features:
        if feature in df.columns:
            categories = set(df[feature].dropna().unique())
            for category in categories:
                count = df[df[feature] == category].shape[0]
                entry = {
                    'feature_name': feature,
                    'category': str(category),
                    'year_03': False,
                    'year_12': False,
                    'count_03': 0,
                    'count_12': 0,
                    'count': count
                }
                result.append(entry)
    
    return pd.DataFrame(result)


def create_category_versions(df, feature_categories, only_12=False):
    """
    Create multiple versions of the dataset with different category replacements.
    
    Args:
        df: pandas DataFrame to modify
        feature_categories: DataFrame with columns [feature_name, category, year_03, year_12]
    
    Returns:
        List of DataFrames, each with one category replacement
    """
    
    # Iterate through each feature and category combination
    for _, row in feature_categories.iterrows():
        new_df = df.copy()
        feature = row['feature_name']
        category = row['category']
        
        # Create feature category identifier
        if only_12:
            feature_cat_id = f"{feature}@{category}_12".replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '').replace(',', '_').replace('/', '_')
        else:
            feature_cat_id = f"{feature}@{category}".replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '').replace(',', '_').replace('/', '_')
        
        # Replace values for year 03 if applicable
        if row['year_03'] and not only_12:
            col_03 = f"{feature}_03"
            if col_03 in new_df.columns:
                new_df[col_03] = new_df[col_03].where(new_df[col_03].isna(), category)
        
        # Replace values for year 12 if applicable
        if row['year_12']:
            col_12 = f"{feature}_12"
            if col_12 in new_df.columns:
                new_df[col_12] = new_df[col_12].where(new_df[col_12].isna(), category)
        
        new_df['feature_category'] = feature_cat_id

        # Save to data/processed/confact
        new_df.to_csv(f'../data/confact/test_features_confact_{feature_cat_id}.csv', index=False)


if __name__ == "__main__":

    test_features = pd.read_csv('../data/raw/test_features.csv')
    train_features = pd.read_csv('../data/raw/train_features.csv')
    test_labels = pd.read_csv('../data/raw/sdoh_test_labels.csv')
    features = pd.concat([train_features, test_features])

    feature_categories = create_feature_category_list(features)
    print(feature_categories)
    feature_categories['max_count'] = feature_categories[['count_03', 'count_12', 'count']].max(axis=1)

    # Remove numeric features
    numeric_features = ['rjob_hrswk', 'rjob_end', 'rearnings', 'searnings', 'hincome',
                        'hinc_business', 'hinc_rent', 'hinc_assets', 'hinc_cap', 'rinc_pension',
                        'sinc_pension', 'a16a', 'a21']
    # Remove age
    feature_categories = feature_categories[~feature_categories['feature_name'].isin(numeric_features)]
    feature_categories = feature_categories[~feature_categories['feature_name'].isin(['age'])]
    # Remove features with max count less than 100
    feature_categories = feature_categories[(feature_categories['max_count'] > 100)].reset_index(drop=True)

    # Usage:
    create_category_versions(test_features, feature_categories)

    # Add income features
    income_features = ['hincome', 'hinc_business', 'hinc_rent', 'hinc_assets', 'hinc_cap', 'rinc_pension', 'sinc_pension']
    for feature in income_features:
        test_features_income = test_features.copy()
        test_features_income[f'{feature}_03'] = test_features[f'{feature}_03'] + test_features[f'{feature}_03'].median()
        test_features_income[f'{feature}_12'] = test_features[f'{feature}_12'] + test_features[f'{feature}_12'].median()
        test_features_income['feature_category'] = f'{feature}@add_median'
        test_features_income.to_csv(f'../data/confact/test_features_confact_{feature}@add_median.csv', index=False)

