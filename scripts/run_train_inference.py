import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from pipeline_runner import PipelineRunner


def main():

    pipeline_runner = PipelineRunner(
        include_advanced_stats=True,
        include_pred_diff=True,
        include_tabpfn=True,
        n_trials=5,
        n_seeds=5,
        filename_results='results_spec_full.pkl',
        load_results=False,
        path_inference_dataset='../data/raw/test_features.csv',
        path_quantiles_tabpfn='../output/quantiles_tabpfn.pkl'
    )
    pipeline_runner.process()

if __name__ == "__main__":
    main()
