import json
import matplotlib.pyplot as plt

def plot_training(path='./saved_models/code2code/checkpoint-3210'):
    with open(path + '/trainer_state.json', 'r') as file:
        trainer_state = json.load(file)
    plt.figure(figsize=(10, 5))
    plt.plot([data['epoch'] for data in trainer_state['log_history'] if 'loss' in data], [data['loss'] for data in trainer_state['log_history'] if 'loss' in data], label='loss')
    plt.plot([data['epoch'] for data in trainer_state['log_history'] if 'eval_loss' in data], [data['eval_loss'] for data in trainer_state['log_history'] if 'eval_loss' in data], label='eval_loss')
    plt.plot([data['epoch'] for data in trainer_state['log_history'] if 'eval_loss' in data], [data['eval_bleu']['bleu'] for data in trainer_state['log_history'] if 'eval_loss' in data], label='bleu')
    plt.plot([data['epoch'] for data in trainer_state['log_history'] if 'eval_loss' in data], [data['eval_codebleu']['codebleu'] for data in trainer_state['log_history'] if 'eval_loss' in data], label='codebleu')
    plt.title('losses')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.savefig(path + '/loss.png')
        
plot_training()
