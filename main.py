import train_ddi
import train_lep
import dill
import os
import datetime
import torch


def experiment(mode):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # log_dir='../logs/'
    log_dir = None
    if log_dir is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join('logs', now)
    else:
        log_dir = os.path.join('logs', log_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if mode == 'lep':
        return train_lep(device, log_dir)
    if mode == 'ddi':
        return train_ddi(device, log_dir)


if __name__ == '__main__':
    experiment(mode='lep')
