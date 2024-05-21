from utils.lep_data import lep_dataset, CollaterLEP
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import time
import dill
import pickle
import argparse
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import torch.optim as optim
from models.IEHGNN import *
import datetime


def train_loop(epoch,
               gcn_model,
               att_model,
               ff_model,
               loader,
               criterion,
               optimizer,
               device,
               scheduler=None):
    gcn_model.train()
    ff_model.train()
    start = time.time()
    losses = []
    for it, (active, inactive) in enumerate(loader):
        labels = active.y.float().to(device)
        active = active.to(device)
        inactive = inactive.to(device)
        optimizer.zero_grad()
        h_active = gcn_model(active.x, active.edge_index,
                             active.edge_attr.view(-1), active.batch)
        h_active, _, _ = att_model(h_active, active.ex_edge_index,
                                   active.pos.unsqueeze_(1),
                                   active.ex_edge_attr.view(-1, 1))
        h_inactive = gcn_model(inactive.x, inactive.edge_index,
                               inactive.edge_attr.view(-1), inactive.batch)
        h_inactive, _, _ = att_model(h_inactive, inactive.ex_edge_index,
                                     inactive.pos.unsqueeze_(1),
                                     inactive.ex_edge_attr.view(-1, 1))
        h1 = global_mean_pool(h_active, active.batch)
        h2 = global_mean_pool(h_inactive, inactive.batch)
        output = ff_model(h1, h2)
        loss = criterion(output, labels)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return np.mean(losses)


@torch.no_grad()
def test_loop(gcn_model, att_model, ff_model, loader, criterion, device):
    gcn_model.eval()
    ff_model.eval()

    losses = []
    y_true = []
    y_pred = []

    for it, (active, inactive) in enumerate(loader):
        labels = active.y.float().to(device)
        active = active.to(device)
        inactive = inactive.to(device)
        h_active = gcn_model(active.x, active.edge_index,
                             active.edge_attr.view(-1), active.batch)
        h_active, _, _ = att_model(h_active, active.ex_edge_index,
                                   active.pos.unsqueeze_(1),
                                   active.ex_edge_attr.view(-1, 1))
        h_inactive = gcn_model(inactive.x, inactive.edge_index,
                               inactive.edge_attr.view(-1), inactive.batch)
        h_inactive, _, _ = att_model(h_inactive, inactive.ex_edge_index,
                                     inactive.pos.unsqueeze_(1),
                                     inactive.ex_edge_attr.view(-1, 1))
        h1 = global_mean_pool(h_active, active.batch)
        h2 = global_mean_pool(h_inactive, inactive.batch)
        output = ff_model(h1, h2)
        loss = criterion(output, labels)
        losses.append(loss.item())
        y_true.extend(labels.tolist())
        y_pred.extend(output.tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    return np.mean(losses), auroc, f1, acc, y_true, y_pred


def train(args, train_loader, val_loader, log_dir):

    hidden = args.n_hid
    device = args.device
    gcn_model = GNN_LEP(args.n_atom_feats, hidden_dim=hidden).to(device)
    att_model = CrossAttentionLayer(input_nf=2 * hidden,
                                    hidden_nf=2 * hidden,
                                    output_nf=2 * hidden,
                                    n_channel=1,
                                    edges_in_d=1).to(device)
    ff_model = MLP_LEP(hidden).to(device)
    best_val_loss = 999
    best_val_auroc = 0

    params = [x for x in gcn_model.parameters()] + [
        x for x in ff_model.parameters()
    ] + [x for x in att_model.parameters()]
    criterion = nn.BCELoss()
    criterion.to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                            lambda epoch: 0.96**(epoch))
    num_epochs = args.n_epochs
    train_losses, val_losses = [], []
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss = train_loop(epoch, gcn_model, att_model, ff_model,
                                train_loader, criterion, optimizer, device,
                                scheduler)
        train_losses.append(train_loss)
        val_loss, auroc, f1, acc, _, _ = test_loop(gcn_model, att_model,
                                                   ff_model, val_loader,
                                                   criterion, device)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            torch.save(
                {
                    'epoch': epoch,
                    'gcn_state_dict': gcn_model.state_dict(),
                    'ff_state_dict': ff_model.state_dict(),
                    'att_state_dict': att_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'best_val_loss': best_val_loss,
                    'best_val_roc': best_val_auroc,
                    'hyperpara': {
                        'lr': lr,
                        'hidden': hidden
                    }
                }, os.path.join(log_dir, f'best_weights.pt'))
            best_val_loss = val_loss
        elapsed = (time.time() - start)
        print(
            'Epoch:{:03d}, Time:{:.3f}s'.format(epoch, elapsed),
            f'Train loss={train_loss:.3f}, Val loss={val_loss:.3f}, Val AUROC={auroc:.3f}'
        )

    return train_losses, val_losses


def test(args, test_loader, device, log_dir):
    hidden = args.n_hid
    gcn_model = GNN_LEP(args.n_atom_feats, hidden_dim=hidden).to(device)
    att_model = CrossAttentionLayer(input_nf=2 * hidden,
                                    hidden_nf=2 * hidden,
                                    output_nf=2 * hidden,
                                    n_channel=1,
                                    edges_in_d=1).to(device)
    ff_model = MLP_LEP(hidden).to(device)
    criterion = nn.BCELoss()
    criterion.to(device)
    cpt = torch.load(os.path.join(log_dir, f'best_weights.pt'))
    gcn_model.load_state_dict(cpt['gcn_state_dict'])
    ff_model.load_state_dict(cpt['ff_state_dict'])
    att_model.load_state_dict(cpt['att_state_dict'])
    test_loss, auroc, f1, acc, y_true_test, y_pred_test = test_loop(
        gcn_model, att_model, ff_model, test_loader, criterion, device)
    print(
        f'Test loss {test_loss:.4f}, Test AUROC {auroc:.4f}, Test f1 {f1:.4f}, Test acc {acc:.4f}'
    )


def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)


def get_datasets(datadir, seed=27):
    with open(datadir, 'rb') as f:
        data = pickle.load(f)
    lens = len(data)
    split = [0.7, 0.2, 0.1]
    if sum(split) != lens:
        split = list(map(lambda x: int(x * lens), split))
    split[-1] = lens - sum(split[:-1])
    n_train, n_val, n_test = split
    np.random.seed(seed)
    idxs = np.random.permutation(lens)
    idx_train = torch.LongTensor(idxs[:n_train])
    idx_val = torch.LongTensor(idxs[n_train:n_train + n_val])
    idx_test = torch.LongTensor(idxs[n_train + n_val:])
    train_dataset = lep_dataset([data[i] for i in idx_train])
    val_dataset = lep_dataset([data[i] for i in idx_val])
    test_dataset = lep_dataset([data[i] for i in idx_test])
    return train_dataset, val_dataset, test_dataset


def get_loader(train_dataset, val_dataset, test_dataset, batch_size=8):
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=CollaterLEP())
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=CollaterLEP())
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=CollaterLEP())
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_atom_feats',
                        '-nd',
                        type=int,
                        default=63,
                        help='num of input features')
    parser.add_argument('--device',
                        '-d',
                        type=str,
                        default='cuda:0',
                        help='device to use')
    parser.add_argument('--n_hid',
                        '-nh',
                        type=int,
                        default=64,
                        help='num of hidden size')
    parser.add_argument('--n_classes',
                        '-nc',
                        type=int,
                        default=1,
                        help='num of interaction types')
    parser.add_argument('--lr',
                        '-lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--n_epochs',
                        '-e',
                        type=int,
                        default=100,
                        help='num of epochs')
    parser.add_argument('--batch_size',
                        '-bs',
                        type=int,
                        default=8,
                        help='batch size')
    parser.add_argument('--load_dataloader',
                        action='store_true',
                        default=False)
    parser.add_argument('--create_data',
                        '-c',
                        action='store_true',
                        default=False)
    parser.add_argument('--log_every_n_epoch',
                        '-loge',
                        default=1,
                        type=int,
                        help='log every n epoch')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
    parser.add_argument('--test_mode',
                        '-t',
                        action='store_true',
                        default=False)

    args = parser.parse_args()
    now = datetime.datetime.now().strftime("%y-%m-%d %H%M%S")
    log_dir = os.path.join('logs', now)
    os.makedirs(log_dir, exist_ok=True)
    datafile = '../../codes/HGNN/lep/lep_2gcat_allweight.pickle'
    train_dataset, val_dataset, test_dataset = get_datasets(datafile)
    train_loader, val_loader, test_loader = get_loader(train_dataset,
                                                       val_dataset,
                                                       test_dataset,
                                                       args.batch_size)
    train_losses, val_losses = train(args,
                                     train_loader,
                                     val_loader,
                                     log_dir=log_dir)
    if args.test_mode:
        test(args, test_loader, args.device, log_dir)
