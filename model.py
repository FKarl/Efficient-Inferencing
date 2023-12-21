import os
import logging
import uuid
from sklearn import metrics
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from util import load_data, MODELS, TASKS, DATASETS, save_metrics

from transformers import get_scheduler, AutoTokenizer, PreTrainedModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from tokenizing import tokenizer as tk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

import pandas as pd
import numpy as np

import torch.nn.functional as F

from platon.Pruner import Pruner

try:
    import wandb

    logging.info("Wandb installed. Tracking is enabled.")
    WANDB_AVAILABLE = True
except ImportError:
    logging.info("Wandb not installed. Skipping tracking.")
    WANDB_AVAILABLE = False


class TransformerModel:
    def __init__(self, model: PreTrainedModel, training_loader: DataLoader = None, validation_loader: DataLoader = None, load_optimizer: bool = False, model_dir: str = "models", task: TASKS = "SectionClassification", output_dir: str = 'models', dataset: DATASETS = 'CiteWorth', model_name: str = 'BERT', optimized_model_name: str = None) -> None:
        self.model = model
        self.model_dir = model_dir
        self.model_name = model_name
        self.task = task
        self.output_dir = output_dir
        self.dataset = dataset
        self.epochs = 1
        self.optimized_model_name = optimized_model_name

        self.uid = uuid.uuid4()
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f'Using {self.torch_device} device.')
        self.optimizer = None
        self.load_optimizer = load_optimizer

        # early stopping
        self.best_eval_loss = float('inf')
        self.epochs_no_improve = 0
        self.max_epochs_stop = 2

        self.model.to(self.torch_device)

    def extract_logits(self, dataloader: DataLoader = None) -> None:
        dataloader = self.training_loader if dataloader is None else dataloader
        self.model.eval()
        all_logits = []
        logging.info('Extracting logits...')
        for batch in tqdm(dataloader, desc=f'Extracting logits', unit='batch', position=0):
            batch = {k: v.to(self.torch_device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                logits = outputs.logits
                all_logits.append(logits.detach().cpu().numpy())
        df = pd.DataFrame(np.concatenate(all_logits))
        path = f'{self.model_dir}/{self.task}/{self.model}/logits.csv'
        df.to_csv(path, index=False)
        logging.info(f'Logits saved to {path}')


    def train(self, epochs: int = 1, dropout: float = 0.1, learning_rate: float = 2e-5, weight_decay: float = 0.01, num_warmup_steps: int = 500) -> AdamW:
        """
        Fine-tune the model.
        :param model: the pretrained model to be fine-tuned
        :param dataloader: an iterable data loader
        :param args: training arguments (and also some other arguments)
        :return: the fine-tuned model
        """
        self.epochs = epochs

        self.model.train()

        if WANDB_AVAILABLE:
            wandb.watch(self.model)

        num_training_steps = epochs * len(self.training_loader)
        # progress_bar = tqdm(range(num_training_steps))

        self.model.classifier.dropout = torch.nn.Dropout(dropout)  # <-----

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if self.load_optimizer:
            logging.info(f'Loading optimizer')
            self.optimizer.load_state_dict(torch.load(f'{self.model_dir}/{self.task}/{self.model}/optimizer.pt'))
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        torch.backends.cudnn.benchmark = True


        for epoch in range(epochs):
            self.model.train()

            for batch in tqdm(self.training_loader, desc=f'Train {epoch + 1}/{epochs}', unit='batch', position=0):
                batch = {k: v.to(self.torch_device) for k, v in batch.items()}

                outputs = self.model(**batch)

                loss = outputs.loss
                logits = outputs.logits

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                if WANDB_AVAILABLE:
                    wandb.log({'training_loss': loss.item()})

            eval_loss, eval_acc, eval_f1_micro, eval_f1_macro, eval_f1_avg = self.eval(epoch)

            logging.info(f"Eval Loss = {eval_loss}")

            if self.early_stopping(eval_loss, epoch + 1):
                break

            lr_scheduler.step()

    def early_stopping(self, eval_loss: float, epoch: int) -> bool:
        cache_path = f'cache/{self.uid}_best_model_cache.pt'

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.epochs_no_improve = 0
            logging.info(f"No early stopping: Saving model at epoch {epoch} under {cache_path}")
            torch.save(self.model.state_dict(), cache_path)
            logging.info(f"Saving optimizer at epoch {epoch} under {cache_path}")
        else:
            self.epochs_no_improve += 1
            logging.info(f"Early stopping: Epochs with no improvement: {self.epochs_no_improve}")
            if self.epochs_no_improve == self.max_epochs_stop:
                logging.info(f"Early stopping at epoch {epoch}")
                self.model.load_state_dict(torch.load(cache_path))
                if os.path.exists(cache_path):
                    logging.debug(f"Removing cache file {cache_path}")
                    os.remove(cache_path)
                return True

        if epoch == self.epochs and os.path.exists(cache_path):
            logging.debug(f"Removing cache file {cache_path}")
            os.remove(cache_path)
        return False

    def eval(self, epoch: int = 0, dataloader: DataLoader = None, threshold: float = 0.2) -> float:
        dataloader = self.validation_loader if dataloader is None else dataloader
        titels = ['conclusion', 'related work', 'introduction', 'result', 'discussion', 'experiment', 'method']
        self.model.eval()
        predictions = []
        labels = []
        running_loss = 0.0
        logging.info('Evaluating...')
        for batch in tqdm(dataloader, desc=f'Eval {epoch}', unit='batch', position=0):
            batch = {k: v.to(self.torch_device, non_blocking=True) for k, v in batch.items()}
            input_ids = batch['input_ids']
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                running_loss += loss.item() * input_ids.size(0)
                if self.task == 'CiteWorthiness':
                    # print(outputs.logits)
                    # print(outputs.logits.shape)
                    predictions.extend(outputs.logits.argmax(-1).cpu().numpy().tolist())
                    labels.extend(batch['labels'].cpu().numpy().tolist())
                elif self.task == 'SectionClassification':
                    # predict class if sigmoid > threshold
                    predictions.extend(
                        (outputs.logits.sigmoid() > threshold).to(torch.int).cpu().numpy().tolist())
                    labels.extend(batch['labels'].cpu().numpy().tolist())


        if self.task == 'CiteWorthiness':
            assert len(np.unique(labels)) <= 2, "Labels should only contain 0s and 1s for binary classification"
            assert len(np.unique(predictions)) <= 2, "Predictions should only contain 0s and 1s for binary classification"

        elif self.task == 'SectionClassification':
            assert len(np.array(labels).shape) == 2, "Labels should be a 2D array for multilabel classification"
            assert len(np.array(predictions).shape) == 2, "Predictions should be a 2D array for multilabel classification"
            assert np.any(np.sum(labels, axis=1) > 0), "Each label should have at least one positive class"
        # END DEBUG CODE

        accuracy = metrics.accuracy_score(labels, predictions)
        if self.task == 'SectionClassification':
            f1_score_avg = metrics.f1_score(labels, predictions, average='samples')
        else:
            f1_score_avg = metrics.f1_score(labels, predictions, average='binary')
        f1_score_micro = metrics.f1_score(labels, predictions, average='micro')
        f1_score_macro = metrics.f1_score(labels, predictions, average='macro')

        logging.info(f"Accuracy Score = {accuracy}")
        logging.info(f"F1 Score (Samples/Binary) = {f1_score_avg}")
        logging.info(f"F1 Score (Micro) = {f1_score_micro}")
        logging.info(f"F1 Score (Macro) = {f1_score_macro}")

        eval_loss = running_loss / len(dataloader.dataset)

        if WANDB_AVAILABLE:
            wandb.log({'eval_loss': eval_loss, 'eval_acc': accuracy, 'eval_f1_micro': f1_score_micro,
                       'eval_f1_macro': f1_score_macro})

        if self.task == 'SectionClassification':
            classification_report = metrics.classification_report(
                labels,
                predictions,
                output_dict=False,
                target_names=titels,
                digits=4)
            if WANDB_AVAILABLE:
                wandb.log({'classification_report': classification_report})
            logging.info("--- Classification Report: ---")
            logging.info(classification_report)
            #logging.info("--- LATEX: ---")
            #logging.info(tabulate(classification_report, headers='keys', tablefmt='latex'))
        else:
            cm = metrics.confusion_matrix(labels, predictions)
            logging.info("--- Confusion Matrix: ---")
            logging.info(cm)

            precision = metrics.precision_score(labels, predictions, average=None)
            recall = metrics.recall_score(labels, predictions, average=None)
            f1 = metrics.f1_score(labels, predictions, average=None)
            logging.info("--- Precision, Recall, F1 ---")
            logging.info(f'{str(precision)}, {str(recall)},  {str(f1)}')
            f1_score_avg = [precision, recall, f1]

        return eval_loss, accuracy, f1_score_micro, f1_score_macro, f1_score_avg


    def distill(self, teacher_model: PreTrainedModel, teacher_dataloader: DataLoader ,epochs: int = 1, temperature: float = 2.0, dropout: float = 0.1, learning_rate: float = 2e-5, weight_decay: float = 0.01, num_warmup_steps: int = 500, distillation_alpha: float = 0.5) -> None:
        print(">>> Starting distillation...")
        #student_model = self.model
        #student_dataloader = self.training_loader

        self.epochs = epochs
        
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        num_training_steps = epochs * len(self.training_loader)

        self.model.classifier.dropout = torch.nn.Dropout(dropout)  # <-----

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if self.load_optimizer:
            optimizer_path = f'{self.model_dir}/{self.task}/{self.optimized_model_name}/optimizer.pt'
            logging.info(f'Loading optimizer from {optimizer_path}')
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        length = len(teacher_dataloader) if len(teacher_dataloader) < len(self.training_loader) else len(self.training_loader)

        teacher_model.to(self.torch_device)
        #self.model.to(self.torch_device)

        teacher_model.eval()
        self.model.train()

        for epoch in range(epochs):
            teacher_model.eval()
            self.model.train()
            for teacher_batch, student_batch in tqdm(zip(teacher_dataloader, self.training_loader), desc=f'Distill {epoch + 1}/{epochs}', unit='batch', position=0, total=length):
                teacher_inputs = {k: v.to(self.torch_device) for k, v in teacher_batch.items()}
                student_inputs = {k: v.to(self.torch_device) for k, v in student_batch.items()}
                assert torch.equal(teacher_inputs["labels"], student_inputs["labels"]), "Input labels are not equal"
                
                # Compute teacher outputs
                with torch.no_grad():
                    teacher_logits = teacher_model(**teacher_inputs).logits
                
                student_out = self.model(**student_inputs)
                student_logits = student_out.logits

                # Soft target loss / distillation loss- "[...]soft maxes’ output[...]" arXiv:1503.02531
                # Softened probabilities of the teacher model
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
                # Log probabilities of the student model (KL expexts log probabilities for input and probabilities for target)
                student_probs = F.log_softmax(student_logits / temperature, dim=-1)
                # Kullback-Leibler divergence between the teacher and the student
                #try:
                soft_target_loss = criterion(student_probs, teacher_probs)
                # except RuntimeError as e:
                #     print(student_probs.shape, teacher_probs.shape)
                #     print(student_probs, teacher_probs)
                #     print('>>> This is okay if you are in the last batch')
                #     continue

                # try:
                #ground_truth_labels = student_inputs["labels"]

                #hard_target_loss = F.cross_entropy(student_logits, ground_truth_labels if self.task == "SectionClassification" else ground_truth_labels.squeeze()) 
                hard_target_loss = student_out.loss

                combined_loss = distillation_alpha * hard_target_loss + (1.0 - distillation_alpha) * soft_target_loss
                # except ValueError as e:
                #     logging.warning(f'Ignoring batch with wrong batch size or other error: {e}')
                #     wrong_batch_size_counter += 1
                #     if wrong_batch_size_counter > max_wrong_batch_size:
                #         raise ValueError(f'Wrong batch size {wrong_batch_size_counter} times in a row. Aborting.')
                #     continue


                self.optimizer.zero_grad(set_to_none=True)
                combined_loss.backward()
                self.optimizer.step()
                

            print(f"Epoch: {epoch+1}, Loss: {combined_loss.item()}")
            eval_loss, eval_acc, eval_f1_micro, eval_f1_macro, eval_f1_avg = self.eval(epoch)

            logging.info(f"Eval Loss = {eval_loss}")

            if self.early_stopping(eval_loss, epoch + 1):
                break
            lr_scheduler.step()

    def save(self, prefix: str = '') -> None:
        """Save model."""
        logging.info(f'Saving model to {self.output_dir}/{self.task}/{prefix}{self.model_name}-{self.dataset}-{self.epochs}epochs-{self.uid}')
        self.model.save_pretrained(f'{self.output_dir}/{self.task}/{prefix}{self.model_name}-{self.dataset}-{self.epochs}epochs-{self.uid}')

        logging.info(f'Saving tokenizer to {self.output_dir}/{self.task}/{self.model_name}/tokenizer/{tk.tokenizer.name_or_path}')
        tk.tokenizer.save_pretrained(f'{self.output_dir}/{self.task}/{self.model_name}/tokenizer/{tk.tokenizer.name_or_path}')

        logging.info( f'Saving optimizer to {self.output_dir}/{self.task}/{prefix}{self.model_name}-{self.dataset}-{self.epochs}epochs-{self.uid}/optimizer.pt')
        torch.save(self.optimizer.state_dict(),f'{self.output_dir}/{self.task}/{prefix}{self.model_name}-{self.dataset}-{self.epochs}epochs-{self.uid}/optimizer.pt')

    def prune(self, epochs: int = 5, weight_decay: float = 0.01, num_warmup_steps: int = 500, learning_rate: float = 5e-5, use_no_mask: bool = False ) -> None:
        """Prune the model"""
        self.epochs = epochs

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        num_training_steps = epochs * len(self.training_loader)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )

        logging.info( os.path.split("pruned_models")[-1])
        No = os.path.split("pruned_models")[-1].split('_')[0]

        global_step = 1
        #epochs_trained = 0
        steps_trained_in_current_epoch = 0

        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()

        PLATON = Pruner(self.model, total_step=len(self.training_loader), use_no_mask=use_no_mask, pruner_name='PLATON')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(tqdm(self.training_loader, desc=f'Prune {epoch + 1}/{epochs}', unit='batch', position=0)):
                batch = {k: v.to(self.torch_device) for k, v in batch.items()}

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                outputs = self.model(**batch)

                loss = outputs[0]

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % 1 == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    self.optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    threshold, mask_threshold = PLATON.update_and_pruning(self.model, global_step)
                    self.model.zero_grad()
                    global_step += 1


            eval_loss, eval_acc, eval_f1_micro, eval_f1_macro, eval_f1_avg = self.eval(epoch)
            logging.info(f"Eval Loss = {eval_loss} | tr_loss = {tr_loss}")
