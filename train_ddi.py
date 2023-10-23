from utils.ddi_data import My_dataset, CollaterLBA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
from rdkit import Chem

def train_loop(model, loader, optimizer, device):
    model.train()
    loss_all = 0
    total = 0

    for g1,g2 in loader:
        g1 = g1.to(device)
        g2 = g2.to(device)
        # print(data.x.shape)
        optimizer.zero_grad()
        output=model(g1,g2)
        # print('out', output.shape)
        loss = F.binary_cross_entropy_with_logits(output, g1.y)
        loss.backward()
        loss_all += loss.item() #* g1.num_graphs
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
    auroc=0
    for g1,g2 in loader:
        g1 = g1.to(device)
        g2 = g2.to(device)
        output=model(g1,g2)
        # print(output)
        loss = F.binary_cross_entropy_with_logits(output, g1.y)
        # print(loss)
        loss_all += loss.item() #* g1.num_graphs
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
    basedir='./data/zhangDDI/'
    train_df=pd.read_csv(os.path.join(basedir,'ZhangDDI_train.csv'))
    val_df=pd.read_csv(os.path.join(basedir,'ZhangDDI_valid.csv'))
    test_df = pd.read_csv(os.path.join(basedir,'ZhangDDI_test.csv'))
    # df=pd.read_csv(os.path.join(basedir,'ZhangDDI_all=95245.csv'))
    drug_list_df=pd.read_csv( os.path.join(basedir,'drug_list_zhang.csv'))
    df = pd.concat([train_df, val_df, test_df], axis=0)
    df.drop_duplicates(subset=['drugbank_id_1', 'drugbank_id_2'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    all_drugs=drug_list_df.drugbank_id.values
    np.random.seed(25)
    idxs=np.loadtxt('./zhangddi_idxs=20-25.txt',dtype=int)
    old,new=all_drugs[idxs[3,:470]],all_drugs[idxs[3,470:]]
    batch_size=64
    train_df=df[df.drugbank_id_1.isin(old) & df.drugbank_id_2.isin(old)]
    val_test_df=df[df.drugbank_id_1.isin(new) & df.drugbank_id_2.isin(new) | 
      df.drugbank_id_1.isin(new) & df.drugbank_id_2.isin(old) | 
      df.drugbank_id_1.isin(old) & df.drugbank_id_2.isin(new)]
    #从测试集分出部分做验证集
    val_df,test_df=train_test_split(val_test_df,test_size=0.5,random_state=25)
    train_df.reset_index(drop=True,inplace=True)
    val_df.reset_index(drop=True,inplace=True)
    test_df.reset_index(drop=True,inplace=True)
    train_datasets = My_dataset(train_df)
    val_datasets = My_dataset(val_df)
    test_datasets = My_dataset(test_df)
    train_loader = DataLoader(train_datasets, batch_size=batch_size , shuffle=True,drop_last=True,collate_fn=CollaterLBA())
    val_loader=DataLoader(val_datasets, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=CollaterLBA())
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=CollaterLBA())
    hidden_dim=64
    learning_rate=0.001
    num_epochs=30
    # model = GNN_LBA(num_features, hidden_dim=args.hidden_dim).to(device)
    # model = IE_HGNN(27, args.hidden_dim, 1, 0.1)
    model = MLP(hidden_dim)
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
        print(
            'Epoch:{:03d}, Time:{:.3f}s, '.format(epoch, elapsed) +
            'Train Loss:{:.3f}, Test Loss:{:.3f}, auroc:{:.3f}'.format(train_loss, val_loss,auroc))
    if True:
      _,auroc, y_true_test, y_pred_test = test(
          model, test_loader, device)
      print(f"test auroc: {auroc:.3f}") 
      test_file = os.path.join(log_dir, f'ZhangDDI_{auroc:.3f}.pt')
      torch.save({
          'targets': y_true_test,
          'predictions': y_pred_test
      }, test_file)

    return best_val_loss,y_true_test,y_pred_test