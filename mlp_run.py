import uuid
import logging
import os

import numpy as np
import psutil
import scipy.sparse as sp
import torch
import csv

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
from tqdm import trange
from transformers import get_linear_schedule_with_warmup, AutoTokenizer

from mlp_model import MLP, collate_for_mlp, collate_for_mlp_multilabel
from tooling import read_pickle_file, write_pickle_file, read_yaml_file
from util import save_metrics
import time

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def inverse_document_frequency(encoded_docs, vocab_size):
    '''Returns IDF scores in shape [vocab_size]'''
    encoded_docs = encoded_docs.tolist()
    num_docs = len(encoded_docs)
    counts = sp.dok_matrix((num_docs, vocab_size))
    for i, doc in tqdm(enumerate(encoded_docs), desc='Computing IDF'):
        for j in doc:
            counts[i, j] += 1

    tfidf = TfidfTransformer(use_idf=True, smooth_idf=True)
    tfidf.fit(counts)
    return torch.FloatTensor(tfidf.idf_)


class MlpWrapper:
    def __init__(self, runtime_args):
        self.model = None
        self.args = runtime_args
        self.dataset_path = f'datasets/{self.args.dataset}/bert-base-uncased'
        self.uid = uuid.uuid4()
        self.best_eval_loss = float('inf')
        self.epochs_no_improve = 0
        self.max_epochs_stop = 3
        self.idf_path = f'datasets/{self.args.dataset}/mlp-idf'
        if self.args.task == 'SectionClassification':
            self.task_label = 'section_category'
            self.collate_fn = collate_for_mlp_multilabel
            self.num_classes = 7
            self.args.threshold = 0.2
            self.multilabel = True
        else:
            self.task_label = 'label'
            self.collate_fn = collate_for_mlp
            self.num_classes = 2
            self.args.threshold = 0.6
            self.multilabel = False
        # Check and override if threshold is set by user
        self.args.threshold = runtime_args.threshold if runtime_args.threshold is not None else self.args.threshold

    def load_data(self, split_name):
        file_path = f'{self.dataset_path}/{split_name}.pkl'
        dataset_tokenized = read_pickle_file(file_path)

        input_ids = dataset_tokenized['input_ids']
        label_ids = dataset_tokenized[self.task_label].tolist()

        return input_ids, label_ids

    def train(self, train_data):
        train_loader = torch.utils.data.DataLoader(train_data, collate_fn=self.collate_fn, shuffle=True, batch_size=self.args.batch_size, num_workers=self.args.num_workers, pin_memory=('cuda' in str(self.args.device)))

        # len(train_loader) = no of batches
        t_total = len(train_loader) // self.args.gradient_accumulation_steps * self.args.epochs

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Load validation data
        logger.info('Loading data validation during training')
        input_ids, enc_labels = self.load_data('validation')
        validation_data = list(zip(input_ids.tolist(), enc_labels))

        # Train!
        logger.info('***** Running training *****')
        logger.info('\tNum examples = %d', len(train_data))
        logger.info('\tNum Epochs = %d', self.args.epochs)
        logger.info('\tBatch size  = %d', self.args.batch_size)
        logger.info('\tTotal train batch size (w. accumulation) = %d', self.args.batch_size * self.args.gradient_accumulation_steps)
        logger.info('\tGradient Accumulation steps = %d', self.args.gradient_accumulation_steps)
        logger.info('\tTotal optimization steps = %d', t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        train_iterator = trange(self.args.epochs, desc='Epoch')

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_loader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.args.device) for t in batch)
                # Batch: torch.tensor(flat_docs), torch.tensor(offsets), torch.tensor(labels)
                outputs = self.model(batch[0], batch[1], batch[2])
                loss = outputs[0]
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1
                    # if WANDB:
                    #     wandb.log({'epoch': epoch, 'lr': scheduler.get_last_lr()[0], 'loss': loss})

            _eval_acc, eval_loss, _eval_f1_micro, _eval_f1_macro, _eval_f1_samples = self.evaluate(validation_data)
            logger.info(f'Eval loss after epoch {epoch}: {eval_loss}')
            if self.early_stopping(eval_loss, epoch + 1):
                break

        return global_step, tr_loss / global_step

    def early_stopping(self, eval_loss: float, epoch: int) -> bool:
        cache_path = f'{self.uid}_best_model_cache.pt'

        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.epochs_no_improve = 0
            logging.info(f'No early stopping: Saving model at epoch {epoch}')
            torch.save(self.model.state_dict(), cache_path)
        else:
            self.epochs_no_improve += 1
            logging.info(f'Early stopping: Epochs with no improvement: {self.epochs_no_improve}')
            if self.epochs_no_improve == self.max_epochs_stop:
                logging.info(f'Early stopping at epoch {epoch}')
                # override argument for easy metrics saving
                self.args.epoch = epoch
                self.model.load_state_dict(torch.load(cache_path))
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                return True

        if epoch == self.args.epochs and os.path.exists(cache_path):
            logging.debug(f'Removing cache file {cache_path}')
            os.remove(cache_path)
        return False

    def evaluate(self, dev_or_test_data, data_loader=None):
        titles = ['conclusion', 'related work', 'introduction', 'result', 'discussion', 'experiment', 'method']
        if data_loader is None:
            #data_loader = torch.utils.data.DataLoader(dev_or_test_data,collate_fn=self.collate_fn,num_workers=self.args.num_workers,batch_size=self.args.test_batch_size,pin_memory=('cuda' in str(self.args.device)),shuffle=False,)
            data_loader = torch.utils.data.DataLoader(dev_or_test_data, collate_fn=self.collate_fn,
                                                      num_workers=self.args.num_workers,
                                                      batch_size=self.args.test_batch_size,
                                                      pin_memory=True, shuffle=False, )
        all_logits = []
        all_targets = []
        nb_eval_steps, eval_loss = 0, 0.0
        for batch in tqdm(data_loader, desc='Evaluating'):
            self.model.eval()
            batch = tuple(t.to(self.args.device) for t in batch)
            #print(batch)
            with torch.no_grad():
                start_wall_time = time.time()
                #print(f'start_wall_time: {start_wall_time}')
                print(batch[0], batch[1], batch[2])
                outputs = self.model(batch[0], batch[1], batch[2])
                end_wall_time = time.time()
                #print(f'end_wall_time: {end_wall_time}')
                print(f'wall_time: {end_wall_time - start_wall_time}')
                all_targets.append(batch[2].detach().cpu())
            nb_eval_steps += 1
            # outputs [:2] should hold loss, logits
            loss, logits = outputs[:2]
            eval_loss += loss.mean().item()
            all_logits.append(logits.detach().cpu())

        logits = torch.cat(all_logits)
        logits = torch.sigmoid(logits) if self.multilabel else torch.softmax(logits, dim=0)
        logits = logits.numpy()
        targets = torch.cat(all_targets).numpy()
        eval_loss /= nb_eval_steps

        if self.multilabel:
            logits[logits >= self.args.threshold] = 1
            logits[logits < self.args.threshold] = 0
            preds = logits
            acc = accuracy_score(targets, logits)
            f1_samples = f1_score(targets, preds, average='samples')
        else:
            preds = np.argmax(logits, axis=1)
            acc = (preds == targets).sum() / targets.size
            f1_samples = f1_score(targets, preds, average='binary')

        precision = precision_score(targets, preds, average=None)
        recall = recall_score(targets, preds, average=None)

        f1_micro = f1_score(targets, preds, average='micro')
        f1_macro = f1_score(targets, preds, average='macro')

        logging.info(f"Accuracy Score = {acc}")
        logging.info(f"F1 Score (Samples/Binary) = {f1_samples}")
        logging.info(f"F1 Score (Micro) = {f1_micro}")
        logging.info(f"F1 Score (Macro) = {f1_macro}")
        logging.info(f"Precision = {precision}")
        logging.info(f"Recall = {recall}")

        # if WANDB:
        #    wandb.log({'eval_loss': eval_loss, 'eval_acc': acc, 'eval_f1_micro': f1_samples,
        #               'eval_f1_macro': f1_macro})
        
        if self.multilabel:
            classification_report = metrics.classification_report(
                targets,
                preds,
                output_dict=False,
                target_names=titles,
                digits=4)
            # if WANDB:
            #    wandb.log({'classification_report': classification_report})
            logging.info("--- Classification Report: ---")
            logging.info(classification_report)
        else:
            cm = metrics.confusion_matrix(targets, preds)
            logging.info("--- Confusion Matrix: ---")
            logging.info(cm)
        
        # if WANDB:
        #     wandb.log({'test/acc': acc, 'test/loss': eval_loss, 'test/f1_micro': f1_micro, 'test/f1_macro': f1_macro})

        return acc, eval_loss, f1_micro, f1_macro, f1_samples

    def save_model_and_statistics(self, acc, eval_loss, f1_micro, f1_macro, f1_samples):
        metrics = {
            'acc': acc,
            'eval_loss': eval_loss,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'avg_sample_f1': f1_samples
        }

        if not os.path.exists(f'{self.args.model_dir}/{self.args.task}'):
            os.makedirs(f'{self.args.model_dir}/{self.args.task}')

        torch.save(self.model, f'{self.args.model_dir}/{self.args.task}/{self.args.model}-{self.args.dataset}-{self.args.epochs}epochs-{str(self.uid)}.pt')

        # Save train metrics
        save_metrics(metrics, self.args, str(self.uid), 'train')
        logger.info('Model and metrics saved. Terminating.')

    def run_training_and_test(self):
        input_ids, enc_labels = self.load_data('train')
        train_data = list(zip(input_ids.tolist(), enc_labels))

        vocab_size = AutoTokenizer.from_pretrained('bert-base-uncased').vocab_size

        if os.path.exists(f'{self.idf_path}/idf.pkl') or os.path.exists(f'{self.idf_path}/idf.pkl.gz'):
            logging.info('Idf file found. Loading idf scores...')
            idf_result = read_pickle_file(f'{self.idf_path}/idf.pkl')
        else:
            logging.info('Starting IDF calculation...')
            idf_result = inverse_document_frequency(input_ids, vocab_size)

            if not os.path.exists(self.idf_path):
                os.makedirs(self.idf_path)
            write_pickle_file(f'{self.idf_path}/idf.pkl', idf_result)
        print("idf written")

        idf = idf_result.to(self.args.device)

        self.model = MLP(vocab_size=vocab_size, num_classes=self.num_classes, idf=idf, multilabel=self.multilabel)
        self.model.to(self.args.device)

        # if WANDB:
        #     wandb.watch(model, log_freq=self.args.logging_steps)

        self.train(train_data)

        # Run concluding tests
        input_ids, enc_labels = self.load_data('test')
        test_data = list(zip(input_ids.tolist(), enc_labels))

        acc, eval_loss, f1_micro, f1_macro, f1_samples = self.evaluate(test_data)
        logging.info(f'[{self.args.dataset}] Test accuracy: {acc:.4f}, Eval loss: {eval_loss}, F1 micro: {f1_micro}, F1 macro: {f1_macro}, F1 samples: {f1_samples}')
        self.save_model_and_statistics(acc, eval_loss, f1_micro, f1_macro, f1_samples)

    def run_interference(self):
        # Load model
        model_path = f'{self.args.model_dir}/{self.args.task}/{self.args.model}-{self.args.dataset}.pt'

        self.model = torch.load(model_path, map_location=torch.device('cpu'))

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

        label_mapping = read_yaml_file('label_mapping.yaml')
        index2label = {v: k for k, v in label_mapping[self.task_label].items()}

        input_ids, enc_labels = self.load_data('test')
        test_data = list(zip(input_ids.tolist(), enc_labels))

        self.evaluate(test_data)

        # Perform inference
        #with torch.no_grad():
        #    model.eval()  # Set the model in evaluation mode

        #    while True:
        #        user_input = input('Enter input text: ')
        #        if user_input == 'exit':
        #            break
        #        input_tokens = tokenizer.encode(user_input)
        #        it_t = torch.tensor(input_tokens).to(self.args.device)
        #        it_o = torch.tensor([0]).to(self.args.device)

        #        logits = model(it_t, it_o)
        #        logits = torch.sigmoid(logits) if self.multilabel else torch.softmax(logits, dim=1)
        #        logits = logits.cpu()
        #        logits = logits.numpy()
        #        logits[logits >= self.args.threshold] = 1
        #        logits[logits < self.args.threshold] = 0

                # print the labels for which the logits are 1
        #        for logit_arr in logits:
        #            for idx, l in enumerate(logit_arr):
        #                if l == 1:
        #                    print(index2label[idx])

        #    print('Terminating.')

    def optimize_threshold(self):
        # Load Model
        model_path = f'{self.args.model_dir}/{self.args.task}/{self.args.model}-{self.args.dataset}.pt'
        self.model = torch.load(model_path)

        # load data
        input_ids, enc_labels = self.load_data('test')
        test_data = list(zip(input_ids.tolist(), enc_labels))

        data_loader = torch.utils.data.DataLoader(test_data, collate_fn=self.collate_fn,
                                                  num_workers=self.args.num_workers,
                                                  batch_size=self.args.test_batch_size,
                                                  pin_memory=('cuda' in str(self.args.device)), shuffle=False, )

        # Loop over thresholds 0.01 -> 0.9
        # override default threshold
        self.args.threshold = 0.01

        # Open the CSV file for writing, or create a new one if it doesn't exist
        with open('threshold_results.csv', 'w', newline='') as csvfile:
            fieldnames = ['Threshold', 'Accuracy', 'Eval Loss', 'F1 Micro', 'F1 Macro', 'F1 Samples']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write the header row
            writer.writeheader()

            while self.args.threshold <= 0.99:
                acc, eval_loss, f1_micro, f1_macro, _f1_samples = self.evaluate(test_data, data_loader)

                # Create a dictionary with the values to be written to the CSV
                row = {
                    'Threshold': round(self.args.threshold,2),
                    'Accuracy': round(acc,2),
                    'Eval Loss': round(eval_loss,2),
                    'F1 Micro': round(f1_micro,2),
                    'F1 Macro': round(f1_macro,2),
                }
                # Append the row to the CSV file
                writer.writerow(row)
                self.args.threshold += 0.01


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default=None, choices=['CiteWorth', 'SciSen'])
    parser.add_argument('--task', default=None, choices=['SectionClassification', 'CiteWorthiness'], help='Specify task type (SectionClassification or CiteWorthiness)')
    parser.add_argument('--mode', default='train', choices=['train', 'run', 'optimize_threshold'], help='Specifiy mode: train, run or optimize_threshold')
    parser.add_argument('--threshold', type=float, default=None, help='Specify sigmoid/softmax threshold')
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--num_workers', default=min(psutil.cpu_count(logical=False)-1, 15))
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Training config
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=None, help='Batch size for testing (defaults to train batch size)')
    parser.add_argument('--logging_steps', type=int, default=50, help='Log every X updates steps.')
    parser.add_argument('--unfreeze_embedding', dest='freeze_embedding', default=True, action='store_false', help='Allow updating pretrained embeddings')

    # Training Hyperparameters
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight deay if we apply some.')
    parser.add_argument('--warmup_steps', default=0, type=int, help='Linear warmup over warmup_steps.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.',)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')
    parser.add_argument('--stats_and_exit', default=False, action='store_true', help='Print dataset stats and exit.')

    # MLP Params
    parser.add_argument('--mlp_num_layers', default=1, type=int, help='Number of hidden layers within MLP')
    parser.add_argument('--mlp_hidden_size', default=1024, type=int, help='Hidden dimension for MLP')
    parser.add_argument('--mlp_embedding_dropout', default=0.5, type=float, help='Dropout for embedding / first hidden layer')
    parser.add_argument('--mlp_dropout', default=0.5, type=float, help='Dropout for all subsequent layers')
    parser.add_argument('--seed', default=None, help='Random seed for shuffle augment')
    parser.add_argument('--shuffle_augment', type=float, default=0, help='Factor for shuffle data augmentation')

    args = parser.parse_args()
    args.model = 'MLP'
    args.test_batch_size = (args.batch_size if args.test_batch_size is None else args.test_batch_size)

    if args.dataset is None:
        logging.info('ERROR: Specify --dataset either CiteWorth or SciSen')
        exit(1)
    if args.task is None:
        logging.info('ERROR: Specify --task either SectionClassification or CiteWorthiness')
        exit(1)

    # if WANDB:
    #    wandb.init(project='text-clf')
    #    wandb.config.update(args)

    mlp_wrapper = MlpWrapper(args)

    if args.mode == 'train':
        mlp_wrapper.run_training_and_test()
    elif args.mode == 'run':
        mlp_wrapper.run_interference()
    elif args.mode == 'optimize_threshold':
        mlp_wrapper.optimize_threshold()
        pass

