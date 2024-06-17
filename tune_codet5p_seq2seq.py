"""
Thi file was copied from CodeT5/CodetT5+ and slightly modified

Finetune CodeT5+ models on any Seq2Seq LM tasks
You can customize your own training data by following the HF dataset format to cache it to args.cache_data
Author: Yue Wang
Date: June 2023
"""

import os
import pprint
import argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
import codebleu


def run_training(args, model, data, tokenizer):
    print(f"Starting main loop")

    training_args = Seq2SeqTrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        evaluation_strategy='epoch',
        predict_with_generate=True,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=1,

        dataloader_drop_last=True,
        dataloader_num_workers=4,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    bleu_metric = evaluate.load("bleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # labels = [[data] for data in label_ids]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # return metric.compute(predictions=decoded_predictions, references=decoded_labels)
        return {
            'bleu': bleu_metric.compute(predictions=decoded_predictions, references=decoded_labels),
            'codebleu': codebleu.calc_codebleu(predictions=decoded_predictions, references=decoded_labels, lang="c_sharp")
        }


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data['train'],
        compute_metrics=compute_metrics,
        eval_dataset=data['test']
    )

    trainer.train()

    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint")
        model.save_pretrained(final_checkpoint_dir)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')


def load_tokenize_data(args):
    # Load and tokenize data
    if os.path.exists(args.cache_data):
        processed_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(processed_data)} samples')
        return processed_data
    else:
        # Example code to load and process code_x_glue_ct_code_to_text python dataset for code summarization task
        # datasets = load_dataset("code_x_glue_ct_code_to_text", 'python', split="train")
        
        # Instead use our dataset, the data is in the "data" folder
        datasets = load_dataset("code_x_glue_cc_code_to_code_trans")
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):
            # source = [' '.join(ex) for ex in examples["code_tokens"]]
            # target = [' '.join(ex) for ex in examples["docstring_tokens"]]

            source = examples["java"]
            target = examples["cs"]
            
            args.max_source_len = 100
            args.max_target_len = 100

            model_inputs = tokenizer(source, max_length=args.max_source_len, padding="max_length", truncation=True)
            labels = tokenizer(target, max_length=args.max_target_len, padding="max_length", truncation=True)

            model_inputs["labels"] = labels["input_ids"].copy()
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]
            return model_inputs

        print(datasets)
        processed_data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets['train'].column_names,
            num_proc=64,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(processed_data)} samples')
        processed_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return processed_data


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    data = load_tokenize_data(args)

    if args.data_num != -1:
        data['train'] = data['train'].select([i for i in range(args.data_num)])
        data['test'] = data['test'].select([i for i in range(args.data_num)])

    # Load model from `args.load`
    # model = AutoModelForSeq2SeqLM.from_pretrained(args.load, trust_remote_code=True).to('cuda')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.load)
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")

    run_training(args, model, data, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--max-source-len', default=320, type=int)
    parser.add_argument('--max-target-len', default=128, type=int)
    parser.add_argument('--cache-data', default='cache_data/code2code', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-220m', type=str)

    # Training
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=200, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/code2code", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
