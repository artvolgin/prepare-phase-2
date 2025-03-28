import warnings
from data_preprocessor import DataPreprocessor
from difference_predictor import DifferencePredictor
from ensemble_predictor import EnsemblePredictor
from data_builder import DataBuilder
from tabpfn_runner import TabpfnRunner
import pickle
import time
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

class PipelineRunner:
    def __init__(self, 
                 path_data_raw='../data/raw/',
                 path_data_processed='../data/processed/',
                 path_quantiles_tabpfn='../output/quantiles_tabpfn.pkl',
                 path_output='../output/',
                 filename_results='results.pkl',
                 path_models='../models/',
                 path_inference_dataset=None,
                 n_trials=5,
                 n_seeds=5,
                 n_ens_configs=128,
                 quantiles_n=10,
                 random_state=42,
                 drop_random_features=False,
                 replace_outliers=False,
                 remove_na_obs=False,
                 include_advanced_stats=True,
                 include_pred_diff=True,
                 include_tabpfn=True,
                 load_results=False):
        """
        Initialize the pipeline with the specified parameters.
        """
        self.path_data_raw = path_data_raw
        self.path_data_processed = path_data_processed
        self.path_quantiles_tabpfn = path_quantiles_tabpfn
        self.path_output = path_output
        self.filename_results = filename_results
        self.path_models = path_models
        self.n_trials = n_trials
        self.n_seeds = n_seeds
        self.random_state = random_state
        self.drop_random_features = drop_random_features
        self.replace_outliers = replace_outliers
        self.remove_na_obs = remove_na_obs
        self.load_results = load_results
        self.path_inference_dataset = path_inference_dataset
        self.include_advanced_stats = include_advanced_stats
        self.include_pred_diff = include_pred_diff
        self.include_tabpfn = include_tabpfn
        self.n_ens_configs = n_ens_configs
        self.quantiles_n = quantiles_n
        
    def process(self):
        """Execute the complete pipeline from data preprocessing to final predictions.
        
        This method orchestrates the entire pipeline by running the following steps:
        1. Data preprocessing using DataPreprocessor
        2. TabPFN prediction using TabpfnRunner 
        3. Difference prediction using DifferencePredictor
        4. Data building using DataBuilder
        5. Ensemble prediction using EnsemblePredictor

        The method tracks execution time and saves the final prediction results along with pipeline parameters.
        """
        print("Starting pipeline execution...")
        
        # Track execution time
        start_time = time.time()

        # Step 1: Data Preprocessing
        print("\n1. Running data preprocessing...")
        data_preprocessor = DataPreprocessor(
            path_data_raw=self.path_data_raw,
            path_data_processed=self.path_data_processed,
            path_output=self.path_output,
            remove_na_obs=self.remove_na_obs,
            drop_random_features=self.drop_random_features,
            replace_outliers=self.replace_outliers,
            path_inference_dataset=self.path_inference_dataset)
        data_preprocessor.process()
        
        # Step 2: TabPFN Prediction
        print("\n2. Running TabPFN prediction...")
        tabpfn_runner = TabpfnRunner(path_inference_dataset=self.path_inference_dataset,
                                     path_quantiles_tabpfn=self.path_quantiles_tabpfn,
                                     load_results=self.load_results,
                                     n_ens_configs=self.n_ens_configs,
                                     quantiles_n=self.quantiles_n)
        tabpfn_runner.process()

        # Step 3: Difference Prediction Pipeline
        print("\n3. Running difference prediction pipeline...")
        difference_predictor = DifferencePredictor(
            path_data_processed=self.path_data_processed,
            path_data_raw=self.path_data_raw,
            path_quantiles_tabpfn=self.path_quantiles_tabpfn,
            path_models=self.path_models,
            path_output=self.path_output,
            n_trials=self.n_trials,
            n_seeds=self.n_seeds,
            random_state=self.random_state,
            include_tabpfn=self.include_tabpfn,
            load_results=self.load_results,
            path_inference_dataset=self.path_inference_dataset
        )
        difference_predictor.process()
        
        # Step 4: Data Builder
        print("\n4. Running data builder...")
        data_builder = DataBuilder(
            include_advanced_stats=self.include_advanced_stats,
            include_pred_diff=self.include_pred_diff,
            include_tabpfn=self.include_tabpfn,
            path_data_processed=self.path_data_processed,
            path_data_raw=self.path_data_raw,
            path_output=self.path_output,
            path_inference_dataset=self.path_inference_dataset)
        data_builder.process()

        # Step 5: Ensemble Prediction
        print("\n5. Running ensemble prediction...")
        ensemble_predictor = EnsemblePredictor(
            n_trials=self.n_trials,
            n_seeds=self.n_seeds,
            load_results=self.load_results,
            path_data_processed=self.path_data_processed,
            path_data_raw=self.path_data_raw,
            path_quantiles_tabpfn=self.path_quantiles_tabpfn,
            path_output=self.path_output,
            path_models=self.path_models,
            filename_results=self.filename_results,
            path_inference_dataset=self.path_inference_dataset
        )
        prediction_results = ensemble_predictor.process()

        # Add pipeline parameters to prediction results
        pipeline_params = {
            'n_trials': self.n_trials,
            'n_seeds': self.n_seeds,
            'random_state': self.random_state,
            'drop_random_features': self.drop_random_features,
            'replace_outliers': self.replace_outliers,
            'remove_na_obs': self.remove_na_obs,
            'include_advanced_stats': self.include_advanced_stats,
            'include_pred_diff': self.include_pred_diff,
            'include_tabpfn': self.include_tabpfn,
            'path_models': self.path_models,
            'path_data_processed': self.path_data_processed,
            'path_data_raw': self.path_data_raw,
            'path_output': self.path_output,
            'filename_results': self.filename_results,
            'path_inference_dataset': self.path_inference_dataset,
            'path_quantiles_tabpfn': self.path_quantiles_tabpfn
        }
        prediction_results['pipeline_params'] = pipeline_params

        # Save results to pickle
        with open(f'{self.path_output}/{self.filename_results}', 'wb') as f:
            pickle.dump(prediction_results, f)

        # Save final predictions for test data
        test_predictions = prediction_results['test_info']
        test_predictions['predicted_score'] = prediction_results['blended_test']['composite_score']
        test_predictions_filename = f'{self.path_output}/{self.path_inference_dataset.split("/")[-1].replace(".csv", "")}_predictions.csv'
        test_predictions.to_csv(test_predictions_filename, index=False)
        print(f'Final predictions for test data saved to {test_predictions_filename}')
        
        # Calculate and print execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nPipeline execution completed in {execution_time:.2f} seconds")
