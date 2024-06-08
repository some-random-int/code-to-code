from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import time

print(torch.cuda.is_available())

# checkpoint = "Salesforce/codet5p-2b"
# checkpoint = "./saved_models/code2code/final_checkpoint"
checkpoint = 'Salesforce/codet5p-220m'
device = "cuda" # "cuda" for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-220m')
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True).to(device)

encoding = tokenizer("public static void hello_world() { System.out.println(\"Hello World\"); }", return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()

start_time = time.time()
outputs = model.generate(**encoding, max_length=100)
print(time.time() - start_time)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))