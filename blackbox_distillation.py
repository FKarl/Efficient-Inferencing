import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from util import load_data, MODELS, TASKS, DATASETS
from tokenizing import tokenizer as tk
from tqdm import tqdm

class Distiller:
    def __init__(self, teacher_model_name, student_model_name,batch_size=32, lr=1e-5, temperature=2.0, num_epochs=3, device=None, task='SectionClassification', dataset='CiteWorth'):
        tk.teacher_tokenizer = AutoTokenizer.from_pretrained(MODELS[teacher_model_name], use_fast=True)
        tk.student_tokenizer = AutoTokenizer.from_pretrained(MODELS[student_model_name], use_fast=True)
        self.teacher_dataloader = load_data(dataset, 'test-short', task, batch_size, f'datasets/{dataset}/{tk.teacher_tokenizer.name_or_path}')
        self.student_dataloader = load_data(dataset, 'validation-short', task, batch_size, f'datasets/{dataset}/{tk.student_tokenizer.name_or_path}')
        
        print('Loading model...')
        if task == 'SectionClassification':
            self.teacher = AutoModelForSequenceClassification.from_pretrained(MODELS[teacher_model_name], num_labels=9)
            self.student = AutoModelForSequenceClassification.from_pretrained(MODELS[student_model_name], num_labels=self.teacher.config.num_labels)
        elif task == 'CiteWorthiness':
            self.teacher = AutoModelForSequenceClassification.from_pretrained(MODELS[teacher_model_name], num_labels=2)
            self.student = AutoModelForSequenceClassification.from_pretrained(MODELS[student_model_name], num_labels=self.teacher.config.num_labels)
        else:
            raise ValueError(f'Unknown task {task}')

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
        self.optimizer = AdamW(self.student.parameters(), lr=lr)
        self.temperature = temperature
        self.num_epochs = num_epochs

        num_training_steps = self.num_epochs * len(self.teacher_dataloader)
        self.progress_bar = tqdm(range(num_training_steps))

        self.teacher.to(self.device).eval()
        self.student.to(self.device).train()

    def train(self):
        for epoch in range(self.num_epochs):
            for teacher_batch, student_batch in zip(self.teacher_dataloader, self.student_dataloader):
                teacher_inputs = {k: v.to(self.device) for k, v in teacher_batch.items()}
                student_inputs = {k: v.to(self.device) for k, v in student_batch.items()}
                
                # Compute teacher outputs
                with torch.no_grad():
                    teacher_logits = self.teacher(**teacher_inputs).logits
                
                student_logits = self.student(**student_inputs).logits

                # Soft target loss / distillation loss- "[...]soft maxes’ output[...]" arXiv:1503.02531
                # Softened probabilities of the teacher model
                teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
                # Log probabilities of the student model (KL expexts log probabilities for input and probabilities for target)
                student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
                # Kullback-Leibler divergence between the teacher and the student
                soft_target_loss = self.criterion(student_probs, teacher_probs)

                # Hard target loss / cross-entropy with true labels "[...] cross-entropy function between the teacher and the student [...]" arXiv:1503.02531
                # Ground truth labels from the inputs
                ground_truth_labels = student_inputs["labels"]
                # The hard target loss is computed using cross-entropy between the student's logits and the true labels
                hard_target_loss = F.cross_entropy(student_logits, ground_truth_labels)

                # Combined loss
                alpha = 0.5
                combined_loss = alpha * hard_target_loss + (1 - alpha) * soft_target_loss

                self.optimizer.zero_grad()
                combined_loss.backward()
                self.optimizer.step()
                self.progress_bar.update(1)

            print(f"Epoch: {epoch+1}, Loss: {combined_loss.item()}")

    def save_student(self, path):
        self.student.save_pretrained(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, default='SectionClassification')
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='CiteWorth')
    parser.add_argument('--teacher_model', choices=MODELS.keys(), default='BERT')
    parser.add_argument('--student_model', choices=MODELS.keys(), default='BERT')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output_dir', type=str, default='models')
    args = parser.parse_args()


    distiller = Distiller(args.teacher_model , args.student_model, batch_size=args.batch_size, lr=args.lr, temperature=args.temperature, num_epochs=args.num_epochs, device=args.device, task=args.task, dataset=args.dataset)
    distiller.train()
    distiller.save_student(f'{args.output_dir}/{args.task}/distilled-{args.teacher_model}-{args.student_model}')