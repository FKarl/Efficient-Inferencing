# convert a model to onnx format
# https://huggingface.co/transformers/serialization.html#converting-a-model-to-onnx
# give huggingface url as param

from util import MODELS


def convert_to_onnx(output_path: str, model: str, tokenizer: str):
    from transformers.onnx import FeaturesManager, export
    from pathlib import Path
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    feature = "sequence-classification"

    # print(output_path, model, tokenizer)

    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature=feature)
    onnx_config = model_onnx_config(model.config)

    onnx_inputs, onnx_outputs = export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path(output_path),
        device="cpu",
    )

    print(onnx_inputs)
    print(onnx_outputs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='bert-base-uncased', help='model name or path')
    parser.add_argument(
        '--output', type=str, default='models/bert-base-uncased.onnx', help='output file name')
    parser.add_argument(
        '--tokenizer', choices=MODELS.keys(), default='BERT', help='tokenizer name or path')
    args = parser.parse_args()

    convert_to_onnx(args.output, args.model, MODELS[args.tokenizer])
