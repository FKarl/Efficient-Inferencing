import argparse
import logging
import random
import torch
from tokenizing import tokenizer as tk
from utils.tooling import memory_stats
from util import load_data, MODELS, TASKS, DATASETS, save_metrics, load_data_distillation
import yaml


from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from model import TransformerModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

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
    parser.add_argument('--model', choices=MODELS.keys(), default='TinyBERT')
    parser.add_argument('--teacher_model', choices=MODELS.keys(), default='SciBERT')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='distilled_models')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--model_dir', type=str, default='models')
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--load_optimizer', type=bool, default=False)
    parser.add_argument('--optimized_model', type=str)
    parser.add_argument('--optimized_teacher_model', type=str)
    parser.add_argument('--distillation_alpha', type=float, default=0.5)

    args = parser.parse_args()

    #fix_indices(args.dataset, args.teacher_model, args.model)
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
        #config = AutoConfig.from_pretrained(f'{args.model_dir}/{args.task}/{args.optimized_model}/config.json')
        # model = BertForSequenceClassification(config)
        #model = AutoModelForSequenceClassification.from_config(config)
        logging.info(f'Loading {args.model_dir}/{args.task}/{args.optimized_model}/pytorch_model.bin')
        model.load_state_dict(torch.load(f'{args.model_dir}/{args.task}/{args.optimized_model}/pytorch_model.bin'))


    logging.info('Loading data...')
    tk.tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model], use_fast=True)
    tk.teacher_tokenizer = AutoTokenizer.from_pretrained(MODELS[args.teacher_model], use_fast=True)

    logging.info(f'Memory before dataloader: {memory_stats()}')
    train_dataloader, teacher_dataloader = load_data_distillation(args.dataset, 'train', args.task, args.batch_size, f'datasets/{args.dataset}/{tk.tokenizer.name_or_path}', f'datasets/{args.dataset}/{tk.teacher_tokenizer.name_or_path}')
    validation_dataloader = load_data(args.dataset, 'validation', args.task, args.batch_size,
                                      f'datasets/{args.dataset}/{tk.tokenizer.name_or_path}')
    logging.info(f'Memory after dataloader: {memory_stats()}')

    # distill
    logging.info('>> Distillation...')
    model_handler = TransformerModel(model, training_loader=train_dataloader, validation_loader=validation_dataloader, load_optimizer=args.load_optimizer, model_dir=args.model_dir, task=args.task, dataset=args.dataset, output_dir='distilled_models', model_name=args.optimized_model, optimized_model_name=args.optimized_model)


    if args.task == 'SectionClassification':
        teacher_model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.teacher_model], num_labels=7,
                                                                problem_type="multi_label_classification")
    elif args.task == 'CiteWorthiness':
        teacher_model = AutoModelForSequenceClassification.from_pretrained(MODELS[args.teacher_model], num_labels=2,
                                                                problem_type="single_label_classification")
    else:
        raise ValueError(f'Unknown task {args.task}')

    

    if args.optimized_teacher_model:
        #config = AutoConfig.from_pretrained(f'{args.model_dir}/{args.task}/{args.optimized_teacher_model}/config.json')
        logging.info(f'Loading {args.model_dir}/{args.task}/{args.optimized_teacher_model}/pytorch_model.bin')
        teacher_model.load_state_dict(torch.load(f'{args.model_dir}/{args.task}/{args.optimized_teacher_model}/pytorch_model.bin'))

    model_handler.distill(teacher_model=teacher_model, teacher_dataloader=teacher_dataloader, epochs=args.epochs, dropout=args.dropout, learning_rate=args.learning_rate, weight_decay=args.weight_decay, num_warmup_steps=args.num_warmup_steps, distillation_alpha=args.distillation_alpha)

    # save model
    logging.info('Saving model...')
    model_handler.save(prefix='distilled-')

    logging.info('Testing...')
    train_dataloader = None
    validation_dataloader = None
    test_dataloader = load_data(args.dataset, 'test', args.task, args.batch_size, f'datasets/{args.dataset}/{tk.tokenizer.name_or_path}')
    test_loss, test_acc, test_f1_micro, test_f1_macro, test_f1_avg = model_handler.eval(epoch=1,dataloader=test_dataloader)
    metrics = {'test_loss': test_loss, 'test_acc': test_acc, 'test_f1_micro': test_f1_micro, 'test_f1_macro': test_f1_macro}
    if args.task == 'SectionClassification':
        metrics['test_f1_avg'] = test_f1_avg
    save_metrics(metrics, args, model_handler.uid, 'test')

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == '__main__':
    main()
