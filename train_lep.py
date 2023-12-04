from models.IEHGNN import IEHGNN
from utils.ddi_data import My_dataset, CollaterLBA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
from rdkit import Chem
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
import time
import dill


def train_loop(model, loader, optimizer, device):
    model.train()
    loss_all = 0
    total = 0

    for g1, g2 in loader:
        g1 = g1.to(device)
        g2 = g2.to(device)
        # print(data.x.shape)
        optimizer.zero_grad()
        output = model(g1, g2)
        # print('out', output.shape)
        loss = F.binary_cross_entropy_with_logits(output, g1.y.float())
        loss.backward()
        loss_all += loss.item()  #* g1.num_graphs
        # total += g1.num_graphs
        optimizer.step()
    # return loss_all / total
    return loss_all / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    loss_all = 0
    total = 0
    y_true = []
    y_pred = []
    auroc = 0
    for g1, g2 in loader:
        g1 = g1.to(device)
        g2 = g2.to(device)
        output = model(g1, g2)
        # print(output)
        loss = F.binary_cross_entropy_with_logits(output, g1.y.float())
        # print(loss)
        loss_all += loss.item()  #* g1.num_graphs
        # total += g1.num_graphs
        y_true.extend(g1.y.tolist())
        y_pred.extend(output.tolist())
    try:
        auroc = roc_auc_score(y_true, y_pred)
    except:
        # print(y_true,y_pred)
        pass
    # return loss_all / total, auroc, y_true, y_pred
    return loss_all / len(loader.dataset), auroc, y_true, y_pred


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def train(device, log_dir, rep=None, test_mode=False):
    datadir = './data/lep_seed=23_r=4.5_lim=10/'
    train_loader = dill.load(
        open(os.path.join(datadir, 'train_loader.pkl'), 'rb'))
    val_loader = dill.load(open(os.path.join(datadir, 'val_loader.pkl'), 'rb'))
    test_loader = dill.load(
        open(os.path.join(datadir, 'test_loader.pkl'), 'rb'))
    hidden_dim = 64
    learning_rate = 0.0001
    num_epochs = 30
    model = IEHGNN(18, hidden_dim)
    model.to(device)
    best_val_loss = 999
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss = train_loop(model, train_loader, optimizer, device)
        val_loss, auroc, y_true, y_pred = test(model, val_loader, device)
        if val_loss < best_val_loss:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
            # plot_corr(y_true, y_pred, os.path.join(log_dir, f'corr_{split}.png'))
            best_val_loss = val_loss
        elapsed = (time.time() - start)
        print('Epoch:{:03d}, Time:{:.3f}s, '.format(epoch, elapsed) +
              'Train Loss:{:.3f}, Test Loss:{:.3f}, auroc:{:.3f}'.format(
                  train_loss, val_loss, auroc))
    if True:
        _, auroc, y_true_test, y_pred_test = test(model, test_loader, device)
        print(f"test auroc: {auroc:.3f}")
        test_file = os.path.join(log_dir, f'ZhangDDI_{auroc:.3f}.pt')
        torch.save({
            'targets': y_true_test,
            'predictions': y_pred_test
        }, test_file)

    return best_val_loss, y_true_test, y_pred_test
