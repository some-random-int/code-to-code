# code-to-code
We want to look into the domain of machine translation of programming languages. With this project, we aim to train a pre-trained model in for the translation of Java code and C# code. After that, we will build our own model from scratch (using PyTorch or TensorFlow). Both approaches will be evaluated at the end.

## BLEU score
BLEU score of pre-trained code-t5p
```json
{'bleu': 0.27616698787238064, 'precisions': [0.4172661870503597, 0.2961165048543689, 0.23955773955773957, 0.19651741293532338], 'brevity_penalty': 1.0, 'length_ratio': 1.774468085106383, 'translation_length': 834, 'reference_length': 470}
```
BLEU score of 1 epoch optimied model
```json
{'bleu': 0.3360167248253018, 'precisions': [0.48464619492656874, 0.35859269282814615, 0.2962962962962963, 0.24756606397774686], 'brevity_penalty': 1.0, 'length_ratio': 1.5936170212765957, 'translation_length': 749, 'reference_length': 470}
```