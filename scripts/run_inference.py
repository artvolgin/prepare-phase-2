# Libraries
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from pipeline_runner import PipelineRunner

def main():

    pipeline_runner = PipelineRunner(
        include_advanced_stats=True,
        include_pred_diff=True,
        include_tabpfn=True,
        load_results=True, # load results from previous run of run_train_inference.py
        path_inference_dataset=f'new_dataset.csv',
        filename_results=f'inference/results_new_dataset.pkl'
    )
    pipeline_runner.process()


if __name__ == "__main__":
    main()
