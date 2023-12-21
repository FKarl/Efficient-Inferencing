import torch
from tqdm import tqdm
from mlp_model import collate_for_mlp_multilabel, collate_for_mlp
from tooling import read_pickle_file

# instantiate model
torch_model = torch.load('finetuned_models/CiteWorthiness/MLP-SciSen.pt', map_location=torch.device('cpu'))

# load data
file_path = f'datasets/CiteWorth/bert-base-uncased/test.pkl'
dataset_tokenized = read_pickle_file(file_path)

task_label = 'label'
collate_fn = collate_for_mlp

input_ids = dataset_tokenized['input_ids']
label_ids = dataset_tokenized[task_label].tolist()

test_data = list(zip(input_ids.tolist(), label_ids))
test_data = test_data[:1]
test_dataloader = torch.utils.data.DataLoader(test_data, collate_fn=collate_fn, batch_size=1, pin_memory=False, shuffle=False)

torch_input = next(iter(test_dataloader))
print(torch_input)
#torch_in = {'input.1': torch_input[0], 'offsets': torch_input[1], 'target': torch_input[2]}
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
onnx_program = torch.onnx.dynamo_export(torch_model, torch_input[0], torch_input[1], torch_input[2], export_options=export_options)
onnx_program.save('finetuned_models/CiteWorthiness/MLP-SciSen.onnx')