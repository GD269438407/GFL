import numpy as np
import subprocess
from math import sqrt
from sklearn.metrics import average_precision_score
from scipy import stats
import random
import torch
import os
from tqdm import tqdm

device = torch.device('cuda:0')

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/main_cold_target_3_withoutpre')

def train(model, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        pre_y,loss = model(data)
        # 这个过程确保数据中的 NaN 值被适当地处理
        # mask, target = nan2zero_get_mask(data, 'None', self.cfg)
        # cls_loss = self.metric.loss_func(c_logit, target.float(), reduction='none') * mask
        # cls_loss = cls_loss.sum() / mask.sum()
        #
        # mix_f = self.model.mix_cs_proj(c_f, s_f)
        # inv_loss = self.simsiam_loss(c_f, mix_f)

        # loss = loss_fn(output.float(),  data.y.view(-1, 1).float().to(device)).float()

        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
        # if batch_idx % 5 == 0:  # every 1000 mini-batches...
            # ...log the running loss
        # loss_str = loss.item()
        # loss_str = str(loss_str)
        # writer.add_scalar('training loss',
        #                   loss_str,
        #                   epoch)
        # writer.close()
        # writer.close()
def predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_preds_1 = []
    total_labels_1 = []
    total_sm_1 = []
    total_target_1=[]
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader):
            data = data.cuda()
            pre_y_final,output = model(data)
            total_preds = torch.cat((total_preds, pre_y_final.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
            total_preds_1.extend(pre_y_final.cpu().numpy().flatten())
            total_labels_1.extend(data.y.view(-1, 1).cpu().numpy().flatten())
            total_sm_1.extend(data.sm)
            total_target_1.extend(data.target)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(),total_sm_1,total_target_1,total_labels_1,total_preds_1


def pre_train(model, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    loss_fn = torch.nn.MSELoss()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        
        optimizer.zero_grad()
        
        output = model(data)
        loss = loss_fn(output, data[-3].float().to(device))

        loss.backward()
        optimizer.step()
def pre_predicting(model, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader):
            
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data[-3].view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
atom_dict = {5: 'C',
             6: 'C',
             9: 'O',
             12: 'N',
             15: 'N',
             21: 'F',
             23: 'S',
             25: 'Cl',
             26: 'S',
             28: 'O',
             34: 'Br',
             36: 'P',
             37: 'I',
             39: 'Na',
             40: 'B',
             41: 'Si',
             42: 'Se',
             44: 'K',
             }



def get_aupr(Y, P, threshold=7.0):
    # print(Y.shape,P.shape)
    Y = np.where(Y >= 7.0, 1, 0)
    P = np.where(P >= 7.0, 1, 0)
    aupr = average_precision_score(Y, P)
    return aupr
def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI

def ci(Y, P):
    summ = 0
    pair = 0

    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if (Y[i] > Y[j]):
                    pair += 1
                    summ += 1 * (P[i] > P[j]) + 0.5 * (P[i] == P[j])

    if pair is not 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


# def calculate_metrics(Y, P, dataset='kiba'):
#     # aupr = get_aupr(Y, P)
#     cindex = get_cindex(Y, P)  # DeepDTA
#     rm2 = get_rm2(Y, P)  # DeepDTA
#     mse = get_mse(Y, P)
#     pearson = get_pearson(Y, P)
#     spearman = get_spearman(Y, P)
#     rmse = get_rmse(Y, P)
#
#     print('metrics for ', dataset)
#     # print('aupr:', aupr)
#     print('cindex:', cindex)
#
#     print('rm2:', rm2)
#     print('mse:', mse)
#     print('pearson', pearson)
#     print('spearman',spearman)

def calculate_metrics(Y, P, dataset='davis'):
    # aupr = get_aupr(Y, P)
    cindex = ci(Y, P)  # DeepDTA
    rm2 = get_rm2(Y, P)  # DeepDTA
    mse = get_mse(Y, P)
    pearson1 = pearson(Y,P)
    spearman1 = spearman(Y, P)
    rmse1 = rmse(Y, P)

    print('metrics for ', dataset)
    # print('aupr:', aupr)
    print('cindex:', cindex)
    print('rmse:', rmse)
    print('rm2:', rm2)
    print('mse:', mse)
    print('pearson', pearson1)
    print('spearman',spearman1)