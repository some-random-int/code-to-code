from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time
import evaluate
import codebleu


device = "cuda" # "cuda" for GPU usage or "cpu" for CPU usage
checkpoints = [
    'Salesforce/codet5p-220m',
    './saved_models/code2code/final_checkpoint'
]

print(' > loading test data')
test = load_dataset("code_x_glue_cc_code_to_code_trans", split="test")
bleu = evaluate.load("bleu")

for checkpoint in checkpoints:
    # load model and tokenizer
    print(' > loading model', checkpoint)
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m') # might be necessary to use pretrained model here
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                torch_dtype=torch.float16,
                                                trust_remote_code=True).to(device)
    print(' > model is ready')
    print(' > preprocessing of', len(test['id']), 'test data')

    def predict(data):
        encoding = tokenizer(data, return_tensors="pt").to(device)
        encoding['decoder_input_ids'] = encoding['input_ids'].clone()
        outputs = model.generate(**encoding, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions = [predict(data) for data in test['java']]

    print(' > calculating metric')

    result = {
        'bleu': bleu.compute(predictions=predictions, references=[[data] for data in test['cs']]),
        'codebleu': codebleu.calc_codebleu(predictions=predictions, references=test['cs'], lang="c_sharp")
    }
    print(result)
