import subprocess
import argparse
from sklearn.model_selection import ParameterGrid
from globals import TASKS, DATASETS, MODELS

tf_search_space = {
    'batch_size': [128],
    'learning_rate': [1e-5, 3e-5, 5e-5],
    'dropout': [0.1],
    'epochs': [3],
    'weight_decay': [0.01],
    'num_warmup_steps': [500, 1000],
}

mlp_search_space = {
    'batch_size': [16, 32, 64],
    'learning_rate': [0.001, 0.005],
    'dropout': [0.3, 0.5],
    'epochs': [15],
    'weight_decay': [0.01],
    'num_warmup_steps': [500, 1000],
}


def main():
    """Main function."""
    # set model as argument to this script
    print("Starting HPO")
    print(str(MODELS.keys()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS.keys(), default='BERT')
    #parser.add_argument('--dataset', choices=DATASETS.keys(), default='CiteWorth')
    #parser.add_argument('--task', choices=TASKS, default='SectionClassification')
    args = parser.parse_args()

    model = args.model

    print(f'Running HPO for {model}')

    finetuning_script = 'run_finetuning.sh'
    eval_script = 'run_eval.sh'

    # using sklearn's ParameterGrid to iterate over all possible combinations of hyperparameters
    print("hello")
    iterator = 0

    for dataset in DATASETS.keys():
        for task in TASKS:
            for params in ParameterGrid(tf_search_space):
                finetuning_arguments = ['--task', task, '--dataset', dataset, '--model', model, '--log_level', 'DEBUG']
                for k, v in params.items():
                    finetuning_arguments.append(f'--{k}')
                    finetuning_arguments.append(str(v))

                # run finetuning with current hyperparameters

                finetuning_command = ['sbatch', 'run_finetuning.sh'] + finetuning_arguments
                print(f"Running finetuning with {finetuning_arguments}")
                iterator += 1
                with open(f'hpo_out/finetuning_command_.txt', 'a') as f:
                    process = subprocess.Popen(finetuning_command, stdout=subprocess.PIPE)
                    output, error = process.communicate()
                    f.write(f"{output.decode('utf-8')} {finetuning_command}\n")


if __name__ == '__main__':
    main()

# 91 mobileBERT
# 80 BERT
# 95 TinyBERT
