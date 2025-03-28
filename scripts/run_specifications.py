import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from pipeline_runner import PipelineRunner


def main():

    # 1st specification
    print("------------- Running 1st specification... -------------")
    pipeline_runner = PipelineRunner(
        include_advanced_stats=False,
        include_pred_diff=False,
        include_tabpfn=False,
        filename_results='results_spec_1.pkl',
        path_inference_dataset='../data/raw/test_features.csv',
        load_results=False
    )
    pipeline_runner.process()

    # 2nd specification
    print("------------- Running 2nd specification... -------------")
    pipeline_runner = PipelineRunner(
        include_advanced_stats=True,
        include_pred_diff=False,
        include_tabpfn=False,
        filename_results='results_spec_2.pkl',
        path_inference_dataset='../data/raw/test_features.csv',
        load_results=False
    )
    pipeline_runner.process()

    # 3rd specification
    print("------------- Running 3rd specification... -------------")
    pipeline_runner = PipelineRunner(
        include_advanced_stats=True,
        include_pred_diff=True,
        include_tabpfn=False,
        filename_results='results_spec_3.pkl',
        path_inference_dataset='../data/raw/test_features.csv',
        load_results=False
    )
    pipeline_runner.process()

    # Drop random features
    print("------------- Running drop random features specification... -------------")
    pipeline_runner = PipelineRunner(
        drop_random_features=True,
        filename_results='results_drop_random_features.pkl',
        path_inference_dataset='../data/raw/test_features.csv',
        load_results=False
    )
    pipeline_runner.process()

    # Replace outliers
    print("------------- Running replace outliers specification... -------------")
    pipeline_runner = PipelineRunner(
        replace_outliers=True,
        filename_results='results_replace_outliers.pkl',
        path_inference_dataset='../data/raw/test_features.csv',
        load_results=False
    )
    pipeline_runner.process()

    # Remove NA observations
    print("------------- Running remove NA observations specification... -------------")
    pipeline_runner = PipelineRunner(
        remove_na_obs=True,
        filename_results='results_remove_na_obs.pkl',
        path_inference_dataset='../data/raw/test_features.csv',
        load_results=False
    )
    pipeline_runner.process()

    # Full specification
    print("------------- Running full specification... -------------")
    pipeline_runner = PipelineRunner(
        include_advanced_stats=True,
        include_pred_diff=True,
        include_tabpfn=True,
        filename_results='results_spec_full.pkl',
        path_inference_dataset='../data/raw/test_features.csv',
        load_results=False
    )
    pipeline_runner.process()

if __name__ == "__main__":
    main()
