import torch

def save_checkpoint(state, is_best_loss, is_best_acc, filename='checkpoint.h5'):
    if is_best_loss:
        print("Saving best loss model...")
        torch.save(state, 'best_loss_' + filename)
    if is_best_acc:
        print("Saving best accuracy model...")
        torch.save(state, 'best_acc_' + filename)