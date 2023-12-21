from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx


def quantize_onnx(model_path: str, output_path: str, quant_type: str) -> None:
    from pathlib import Path
    from onnxruntime.quantization import quantize_dynamic, QuantType, preprocess
    from onnxruntime.transformers.optimizer import optimize_model
    import onnx
    import subprocess

    # model = onnx.load(model_path)

    # optimize model
    print('Optimizing model...')
    optimizer = optimize_model(model_path,
                               # we only use bert like models here so we can use the bert optimization level
                               model_type='bert',
                               use_gpu=False,
                               only_onnxruntime=False)

    optimized_model_path = f'{model_path}.optimized'
    optimizer.save_model_to_file(optimized_model_path)

    # preprocess model
    print('Preprocessing model...')
    subprocess.run(['python', '-m', 'onnxruntime.quantization.preprocess', '--input', optimized_model_path, '--output',
                    optimized_model_path])

    # quantize model
    print('Quantizing model...')
    quantize_dynamic(
        optimized_model_path,
        output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=['MatMul', 'Attention'],
        per_channel=True,
        reduce_range=True,
        extra_options={'WeightSymmetric': False, 'MatMulConstBOnly': True})


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='models/microsoft/deberta-base', help='model name or path')
    parser.add_argument('--output', type=str, default='models/bert-base-uncased-quantized.onnx',
                        help='output file name')
    parser.add_argument('--quant_type', type=str, default='INT8', help='quantization type')
    args = parser.parse_args()

    quantize_onnx(args.model, args.output, args.quant_type)


if __name__ == '__main__':
    main()
