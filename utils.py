from datasets import load_dataset, load_from_disk
import evaluate
import codebleu
import os

def get_compute_metrics_fn(tokenizer):

    bleu_metric = evaluate.load("bleu")

    # computes bleu and codebleu metric
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        labels = [
            [(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels)
        decoded_predictions = tokenizer.batch_decode(predictions)
        return {
            'bleu': bleu_metric.compute(predictions=decoded_predictions, references=decoded_labels),
            'codebleu': codebleu.calc_codebleu(predictions=decoded_predictions, references=decoded_labels, lang="c_sharp")
        }

    return compute_metrics

def load_tokenize_data(tokenizer, cache_path='./cache_data/code2code', max_source_len=100, max_target_len=100, overwrite=False):
    # Load and tokenize data
    if os.path.exists(cache_path) and not overwrite:
        processed_data = load_from_disk(cache_path)
        print(f'  ==> Loaded {len(processed_data)} datasets')
        return processed_data
    else:
        datasets = load_dataset("code_x_glue_cc_code_to_code_trans")

        def preprocess_function(examples):
            source = examples["java"]
            target = examples["cs"]
            
            model_inputs = tokenizer(source, max_length=max_source_len, padding="max_length", truncation=True)
            labels = tokenizer(target, max_length=max_target_len, padding="max_length", truncation=True)

            model_inputs["labels"] = labels["input_ids"].copy()
            model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in model_inputs["labels"]
            ]
            return model_inputs

        processed_data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets['train'].column_names,
            num_proc=64,
            load_from_cache_file=False,
        )
        print(f'  ==> Loaded {len(processed_data)} datasets')
        processed_data.save_to_disk(cache_path)
        print(f'  ==> Saved to {cache_path}')
        return processed_data

def get_comparison_values(metric):
    data = {
        "Naive copy": {"BLEU": 18.54, "Acc": 0.0, "CodeBLEU": 0.0},
        "PBSMT": {"BLEU": 43.53, "Acc": 12.5, "CodeBLEU": 42.71},
        "Transformer": {"BLEU": 55.84, "Acc": 33.0, "CodeBLEU": 63.74},
        "Roborta (code)": {"BLEU": 77.46, "Acc": 56.1, "CodeBLEU": 83.07},
        "CodeBERT": {"BLEU": 79.92, "Acc": 59.0, "CodeBLEU": 85.10},
    }
    keys = list(data.keys())
    values = list(data.values())
    return (keys, [value[metric] / 100 for value in values])