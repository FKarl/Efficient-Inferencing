# Efficient Inferencing in Language Models for Academic Writing Feedback
This is the code for the paper "Efficient Inferencing in Language Models for Academic Writing Feedback"

[Fabian Karl](https://orcid.org/0009-0008-0079-5604), [Lukas Romer](https://orcid.org/0009-0000-6764-5776), [Young-Keun Choi](https://orcid.org/0009-0001-5823-0007), [Dennis Huber](https://orcid.org/0009-0002-5723-9699)
and [Sebastian Boll](https://orcid.org/0009-0004-7618-2469)

Language models such as ChatGPT currently provide valuable
support for academic writing. Nevertheless, the closed-source
nature of modern models poses a challenge, rendering them inaccessible offline, thereby limiting their use on local machines. In
this study, we employ fine-tuning on multiple language models
for academic paper understanding, focusing on cite-worthiness
detection and section classification. The models are further optimized for fast and efficient inference using distillation, pruning,
and quantization techniques. We evaluate their performance on
three different devices: server CPU, notebook CPU, and System
on Chip (SoC). Our findings underscore the potential and tradeoffs of these optimization strategies. The implemented techniques
provide effective ad-hoc solutions for decreasing inference time
while preserving task performance. Notably, our results reveal
that small, highly optimized models can achieve reasonable inference times, even on commodity hardware, paving the way for
enhanced user experiences and accessibility in real-world applications. The implications of our study extend beyond academic
writing, offering valuable insights for practitioners seeking efficient and practical solutions in deploying language models for
diverse tasks.

## Installation
First you need to clone this repository. Run the following command:
```bash
git clone https://github.com/your-username/2023-project-citesec-main.git
```
Once the command completes, a new folder named `2023-project-citesec-main` will be created. This folder contains the project files.
After cloning the repository, the next step is to install the dependencies listed in the `requirements.txt` file. 
For this, you will need Python installed on your system. If you don't have Python, download and install it from [Python's official website](https://www.python.org/downloads/).

1. Navigate to the root directory of the cloned repository in your CLI. For example:
    ```bash
    cd 2023-project-citesec-main
    ```
2. Ensure that you have `pip` installed (it comes with Python by default). You can check by running:
```pip --version```
3. Install the dependencies by running:
    ```
    pip install -r requirements.txt
    ```
This command reads the `requirements.txt` file and installs all the libraries listed in it. 
After completing these steps, you should have a copy of the project and all the necessary dependencies installed on your system.

## Modules and Scripts
Detailed purpose of key modules and scripts:
- _config files_: Are used for storing parameters and a good reference point for very detailed background on our analysis. 
- `/datasets`: [You can use this to dump the raw datasets. The preprocessing will reference this by default. Its output is also saved here.]
- `/output`: [This is used as storage for any graphical or table-like output produced by the project.]
- `/preprocessing/src`: [We use two distinct datasets in this work: CiteWorth and SciSen. You can get the original datasets and run the preprocessing using these dedicated scripts.]
- `/statistics`: [Scripts for a detailed analysis of the general dataset, labels and tokens.]
- `/tokenizing`: [This handles the tokenizing of the preprocessed datasets, also including our centered truncation method.]
- `/utils`: [General methods used across different scripts.]

## Usage
General hints for getting started with the code.
### Preprocessing
Check if the source of the original datasets match your file system. Run the dedicated script of the dataset you want to process.
### Tokenizing
Check the header of the script. All configurations can be found under _Constants_. Change them to your needs and run the main.
### Finetuning
Make sure you are in the main project directory. Run `finetuning.py` using the necessary arguments. For instance:
```
python finetuning.py --task SectionClassification --dataset CiteWorth --model BERT
```
Arguments:
- `--task`: Task for the model (Default: `SectionClassification`).
- `--dataset`: Dataset to be used (Default: `CiteWorth`).
- `--model`: Machine learning model to use (Default: `BERT`).
- `--epochs`: Number of training epochs (Default: `10`).
- `--batch_size`: Configures the batch size for training (Default: `8`).
- `--learning_rate`: Specifies the learning rate (Default: `2e-5`).
- `--weight_decay`: Sets the weight decay factor (Default: `0.01`).
- `--num_warmup_steps`: Number of warmup steps for the learning rate scheduler (Default: `0`).
- `--output_dir`: Defines the directory for saving output files (Default: `models`).
- `--log_level`: Sets the logging level. Options include `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (Default: `INFO`).
- `--seed`: Provides a seed value for random number generation to ensure reproducibility (Default: `1337`).
- `--evaluate`: Boolean flag to control whether to evaluate the model after training (Default: `True`).
- `--dropout`: Sets the dropout rate for the model (Default: `0.1`).
- `--model_dir`: Specifies the directory where the model files are stored (Default: `models`)
- `--threshold`: Sets a threshold value for certain model operations (Default: `0.2`)
- `--load_optimizer`: Boolean flag to indicate whether to load an optimizer state (Default: `False`)
- `--optimized_model`: Path to an optimized model file, if any.
- `--safe_logits`: Boolean flag to enable or disable safe logits. (Default: `False`)

Remember to replace placeholders with actual values as per your requirement.


### Hyperparameter Optimization
Make sure you are in the main project directory. Run `finetuning.py` using the necessary arguments. For instance:
```
python finetuning.py --model BERT
```
Arguments:
- `--model`: Machine learning model to use (Default: `BERT`).

### Distillation
Make sure you are in the main project directory. Run `distillation.py` using the necessary arguments. For instance:
```
python distillation.py --task SectionClassification --dataset CiteWorth --model TinyBERT --distillation_alpha 0.5
```
Arguments:
- `--task`: Task for the model (Default: `SectionClassification`).
- `--dataset`: Dataset to be used (Default: `CiteWorth`).
- `--model`: Machine learning model to use for student (Default: `TinyBERT`).
- `--teacher_model`: Machine learning model to use for teacher (Default: `SciBERT`).
- `--epochs`: Number of training epochs (Default: `10`).
- `--batch_size`: Configures the batch size for training (Default: `8`).
- `--learning_rate`: Specifies the learning rate (Default: `2e-5`).
- `--weight_decay`: Sets the weight decay factor (Default: `0.01`).
- `--num_warmup_steps`: Number of warmup steps for the learning rate scheduler (Default: `0`).
- `--output_dir`: Defines the directory for saving output files (Default: `distilled_models`).
- `--log_level`: Sets the logging level. Options include `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (Default: `INFO`).
- `--seed`: Provides a seed value for random number generation to ensure reproducibility (Default: `1337`).
- `--evaluate`: Boolean flag to control whether to evaluate the model after training (Default: `True`).
- `--dropout`: Sets the dropout rate for the model (Default: `0.1`).
- `--model_dir`: Specifies the directory where the model files are stored (Default: `models`).
- `--threshold`: Sets a threshold value for certain model operations (Default: `0.2`).
- `--load_optimizer`: Boolean flag to indicate whether to load an optimizer state (Default: `False`).
- `--optimized_model`: Path to an optimized model file for the student, if any.
- `--optimized_teacher_model`: Path to an optimized model file for the teacher, if any.
- `--distillation_alpha`: Alpha value for distillation process (Default: `0.5`).

### Pruning
Make sure you are in the main project directory. Run `pruning.py` using the necessary arguments. For instance:
```
python pruning.py --task SectionClassification --dataset CiteWorth --model TinyBERT --distillation_alpha 0.5
```
Arguments:
- `--task`: Task for the model (Default: `SectionClassification`).
- `--dataset`: Dataset to be used (Default: `CiteWorth`).
- `--model`: Machine learning model to use (Default: `BERT`).
- `--epochs`: Number of training epochs (Default: `3`).
- `--batch_size`: Configures the batch size for training (Default: `32`).
- `--learning_rate`: Specifies the learning rate (Default: `2e-5`).
- `--weight_decay`: Sets the weight decay factor (Default: `0.01`).
- `--num_warmup_steps`: Number of warmup steps for the learning rate scheduler (Default: `0`).
- `--output_dir`: Defines the directory for saving output files (Default: `pruned_models`).
- `--log_level`: Sets the logging level. Options include `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (Default: `INFO`).
- `--seed`: Provides a seed value for random number generation to ensure reproducibility (Default: `1337`).
- `--evaluate`: Boolean flag to control whether to evaluate the model after training (Default: `True`).
- `--dropout`: Sets the dropout rate for the model (Default: `0.1`).
- `--model_dir`: Specifies the directory where the model files are stored (Default: `models`).
- `--threshold`: Sets a threshold value for certain model operations (Default: `0.2`).
- `--load_optimizer`: Boolean flag to indicate whether to load an optimizer state (Default: `False`).
- `--optimized_model`: Path to an optimized model file, if any.
- `--safe_logits`: Boolean flag to enable or disable safe logits (Default: `False`).
- `--platon_no_mask`: Boolean flag to enable or disable the Platon no mask feature (Default: `False`).

### Quantization
Make sure you are in the main project directory. Run `hyperparameter_optimization.py` using the necessary arguments. For instance:
```
python quantize_onnx.py --task SectionClassification --dataset CiteWorth --model TinyBERT --distillation_alpha 0.5
```
Arguments:
- `--model`: Specifies the name or path of the model to be used (Default: `models/microsoft/deberta-base`).
- `--output`: Sets the file name for the output, typically used for specifying the output model file (Default: `models/bert-base-uncased-quantized.onnx`).
- `--quant_type`: Defines the type of quantization to be applied (Default: `INT8`).

### Measuring Inference Time
Make sure you are in the main project directory. Run `hyperparameter_optimization.py` using the necessary arguments. For instance:
```
python inference_onnx.py --task SectionClassification --dataset CiteWorth --model TinyBERT --distillation_alpha 0.5
```
Arguments:
- `--task`: Specifies the task to be performed by the model (Default: `SectionClassification`).
- `--dataset`: Selects the dataset for the model (Default: `CiteWorth`).
- `--model`: Determines the machine learning model to be used (Default: `BERT`).
- `--number_sentences`: Sets the number of sentences to be processed (Default: `1000`).
- `--optimized_model`: Path to an optimized model file, if available.
- `--token_length`: Specifies the token length for processing (Default: `30`)

### Optional: WandB Integration
If you wish to use WandB for tracking experiments, ensure WandB is installed and configured.

## Authors and Contact
- [Fabian Karl](https://orcid.org/0009-0008-0079-5604)
- [Lukas Romer](https://orcid.org/0009-0000-6764-5776)
- [Young-Keun Choi](https://orcid.org/0009-0001-5823-0007)
- [Dennis Huber](https://orcid.org/0009-0002-5723-9699)
- [Sebastian Boll](https://orcid.org/0009-0004-7618-2469)

## License
The code is licensed under the MIT license.