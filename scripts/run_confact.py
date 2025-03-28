# Libraries
import warnings
import os
# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

from pipeline_runner import PipelineRunner


def main():

    # Run for all files in confact folder
    list_files = os.listdir('../data/confact/')
    for file in list_files:
        try:
            if file.endswith('.csv'):
                print(f"Running {file}...")
            pipeline_runner = PipelineRunner(
                include_advanced_stats=True,
                include_pred_diff=True,
                include_tabpfn=True,
                filename_results=f'confact/results_confact_{file.split(".")[0]}.pkl',
                load_results=True,
                path_inference_dataset=f'../data/confact/{file}'
            )
            pipeline_runner.process()

        except Exception as e:
            print(f"Error running {file}: {e}")


if __name__ == "__main__":
    main()
