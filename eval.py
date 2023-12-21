import argparse
import logging
import random
import torch
from tokenizing import tokenizer as tk
from utils.tooling import memory_stats
from util import load_data, MODELS, TASKS, DATASETS, save_metrics

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification, AutoConfig

from model import TransformerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logging.info('eval.py')

try:
    import wandb

    logging.info("Wandb installed. Tracking is enabled.")
    WANDB_AVAILABLE = True
except ImportError:
    logging.info("Wandb not installed. Skipping tracking.")
    WANDB_AVAILABLE = False

def main():
    """Main function."""
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, default='SectionClassification')
    parser.add_argument('--dataset', choices=DATASETS.keys(), default='CiteWorth')
    parser.add_argument('--model', choices=MODELS.keys(), default='BERT')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--load_optimizer', type=bool, default=False)
    parser.add_argument('--optimized_model', type=str)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if WANDB_AVAILABLE:
        wandb.init(project="DataScienceProjectSoSe2023",
                   entity="fkarl",
                   config=args)

    # load model
    logging.info('Loading model...')
    
    if args.task == 'SectionClassification':
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.model], num_labels=7,
                                                                problem_type="multi_label_classification")
    elif args.task == 'CiteWorthiness':
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.model], num_labels=2,
                                                                problem_type="single_label_classification")
    else:
        raise ValueError(f'Unknown task {args.task}')

    if args.optimized_model:
        # logging.info(f'Loading {args.model_dir}/{args.task}/{args.optimized_model}/pytorch_model.bin')
        # model.load_state_dict(torch.load(f'{args.model_dir}/{args.task}/{args.optimized_model}/pytorch_model.bin'))
        config = AutoConfig.from_pretrained(f'{args.model_dir}/{args.task}/{args.optimized_model}/config.json')
        # model = BertForSequenceClassification(config)
        model = AutoModelForSequenceClassification.from_config(config)
        logging.info(f'Loading {args.model_dir}/{args.task}/{args.optimized_model}/pytorch_model.bin')
        model.load_state_dict(torch.load(f'{args.model_dir}/{args.task}/{args.optimized_model}/pytorch_model.bin'), strict=False)

    tk.tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model], use_fast=True)

    logging.info('Testing...')
    model_handler = TransformerModel(model, load_optimizer=args.load_optimizer, model_dir=args.model_dir, task=args.task, dataset=args.dataset, output_dir=args.output_dir)

    test_dataloader = load_data(args.dataset, 'test', args.task, args.batch_size, f'datasets/{args.dataset}/{tk.tokenizer.name_or_path}')
    test_loss, test_acc, test_f1_micro, test_f1_macro, test_f1_avg = model_handler.eval(epoch=1,dataloader=test_dataloader)
    metrics = {'test_loss': test_loss, 'test_acc': test_acc, 'test_f1_micro': test_f1_micro, 'test_f1_macro': test_f1_macro}
    if args.task == 'SectionClassification':
        metrics['test_f1_avg'] = test_f1_avg
    else:
        metrics['precision/recall/f1'] = test_f1_avg
    save_metrics(metrics, args, model_handler.uid, 'test')

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()
