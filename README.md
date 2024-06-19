# code-to-code
We want to look into the domain of machine translation of programming languages. With this project, we aim to train a pre-trained model in for the translation of Java code and C# code. After that, we will build our own model from scratch (using PyTorch or TensorFlow). Both approaches will be evaluated at the end.

## Dataset
[`code_x_glue_cc_code_to_code_trans`](https://huggingface.co/datasets/google/code_x_glue_cc_code_to_code_trans)

## Performance
Java to C#:
|     Method     |    BLEU    | Acc (100%) |  [CodeBLEU](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Code/code-to-code-trans/CodeBLEU.MD) |  
|    ----------  | :--------: | :-------:  | :-------: |
| Naive copy     |   18.54    |    0.0     |      -    |
| PBSMT      	 |   43.53    |   12.5     |   42.71   |
| Transformer    |   55.84    |   33.0     |   63.74   |
| Roborta (code) |   77.46    |   56.1     |   83.07   |
| CodeBERT   	 | **79.92**  | **59.0**   | **85.10** |


## BLEU score
BLEU score of pre-trained code-t5p
```json
{'bleu': 0.27616698787238064, 'precisions': [0.4172661870503597, 0.2961165048543689, 0.23955773955773957, 0.19651741293532338], 'brevity_penalty': 1.0, 'length_ratio': 1.774468085106383, 'translation_length': 834, 'reference_length': 470}
```
BLEU score of 1 epoch optimied model
```json
{'bleu': 0.3360167248253018, 'precisions': [0.48464619492656874, 0.35859269282814615, 0.2962962962962963, 0.24756606397774686], 'brevity_penalty': 1.0, 'length_ratio': 1.5936170212765957, 'translation_length': 749, 'reference_length': 470}
```