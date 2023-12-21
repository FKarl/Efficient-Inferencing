import yaml
import os

import convert_to_onnx
import run_onnx

# Load models from YAML file


def convert_and_run(model: str, tokenizer: str):



    out_path=f'models/{model}'

    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Convert model to ONNX
    if not os.path.exists(f'{out_path}/model.onnx'):
        print(f'Converting {model} to ONNX...')
        convert_to_onnx.convert_to_onnx(output_path=out_path, model=model, tokenizer=tokenizer)
    else:
        print(f'ONNX model for {model} already exists, skipping conversion...')

    # Run model on ONNX
    print(f'Running {model} on ONNX...')
    print(f'Using tokenizer {tokenizer}')
    run_onnx.run_onnx(model_path=f'models/{model}', tokenizer=tokenizer, serialize=True)
    

if __name__ == '__main__':
    with open('tokenizing/config/models.yaml', 'r') as f:
        models = yaml.safe_load(f)

    models.pop("DeBERTa_large")  # Remove if you want DeBERTa_large to benchmark
	
    for model_name, model_links in models.items():
        print(model_name)
        convert_and_run(*model_links.values())