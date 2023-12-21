from contextlib import contextmanager
from time import time
from tqdm import trange
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
import numpy as np
import yaml, os


mock_text_average = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam volu."
mock_text_max = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea tak."

def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session


@contextmanager
def track_infer_time(buffer: [int]):
    start = time()
    yield
    end = time()

    buffer.append(end - start)


def serialize_results(new_results: dict) -> None:
    existing_results = {}

    if os.path.exists('results.yaml'):
        with open('results.yaml', 'r') as f:
            existing_results = yaml.safe_load(f)

    results = {**existing_results, **new_results}

    with open('results.yaml', 'w') as f:
        yaml.dump(results, f)

def run_onnx(model_path: str, tokenizer: str, serialize: bool) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    cpu_model = create_model_for_provider(model_path, "CPUExecutionProvider")

    # Inputs are provided through numpy array with dtype int64

    model_inputs = tokenizer( mock_text_max, return_tensors="np", truncation=True)

    # Get the input names required by the model
    input_names = cpu_model.get_inputs()
    input_names = [input.name for input in input_names]

    # Filter the model_inputs dictionary to only include the required inputs
    model_inputs = {k: v for k, v in model_inputs.items() if k in input_names}

    # convert to tensor(int64)
    model_inputs = {k: v.astype('int64') for k, v in model_inputs.items()}

    time_buffer = []
    for _ in trange(100):
        with track_infer_time(time_buffer):
            # Run the model (None = get all the outputs)
            out = cpu_model.run(None, input_feed=model_inputs)

    avg_inference_time = sum(time_buffer) / len(time_buffer)
    std_dev_inference_time = np.std(time_buffer)

    print(f"Average inference time: {avg_inference_time}")
    print(f"Standard deviation of inference time: {std_dev_inference_time}")

    results = {
        str(model_path) : {
            'average_inference_time': float(avg_inference_time),
            'standard_deviation_inference_time': float(std_dev_inference_time),
            'runs': len(time_buffer),
        }
    }
    
    serialize_results(results) if serialize else None



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='./models/bert-base-uncased.onnx')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased')
    parser.add_argument('--serialize', type=bool, default=False)

    args = parser.parse_args()

    run_onnx(args.model_path, args.tokenizer, args.serialize)

    

if __name__ == '__main__':
    main()
