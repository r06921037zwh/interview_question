# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:48:37 2020

@author: zhewei
"""
import pandas as pd
import numpy as np
import random, os, argparse
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
import seaborn as sns
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix

from dataset import interviewDataset
from net import net

avg_train_loss = []
avg_val_loss = []
avg_test_loss = []

# create directory for storing figures
directory = 'figures'
if not os.path.exists(directory):
    print("Create dir: {}".format(directory))
    os.makedirs(directory)
        
# read in data and prepare for training
def dataProcess(filename='interviewData.csv', test_percent=0.3):  
    # read csv file
    df = pd.read_csv(filename)
    df.head()
    # make target
    y_data = [int(i == 1) for i in list(df['hardbin_FT1'])]
    y_data = np.array(y_data, 'float64')
    
    # sample train and test dataset
    neg_index = []
    pos_index = []
    for i, y in enumerate(y_data):
        if y != 1:
            neg_index.append(i)
        else:
            pos_index.append(i)
            
    test_pos_index = random.sample(pos_index, int(test_percent*len(pos_index)))
    test_neg_index = random.sample(neg_index, int(test_percent*len(neg_index)))
    train_pos_index = [i for i in pos_index if i not in test_pos_index]
    train_neg_index = [i for i in neg_index if i not in test_neg_index]

    # drop unnecessary features
    df = df.drop(['totaldietestseconds_WS1','wafername_WS1','ecid', 'hardbin_FT1'], 1)
    X_data = df.to_numpy()
    
    # use random forest to rank feature importance
    # use only top 200 
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X_data, y_data)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    #print(len(importances[importances >= 0.002]))
    indices = indices[:200]
    X_data = X_data[:,indices]
    
    
    # perform dimension reduction 
    pca_transformer = PCA(n_components=50)
    X_data = pca_transformer.fit_transform(X_data)
    
    # Feature Normalization.
    # All features should have the same range of values (-1,1)
    sc = StandardScaler()
    X_data = sc.fit_transform(X_data)
    
    # make train and test data
    X_train, y_train = [], []
    X_test, y_test = [], []
    for i, v in enumerate(X_data):
        if i in train_pos_index or i in train_neg_index:
            X_train.append(X_data[i, :])
            y_train.append(y_data[i])
        elif i in test_pos_index or i in test_neg_index:
            X_test.append(X_data[i,:])
            y_test.append(y_data[i])
            
    print("Original pos/neg in data {}/{}".format(sum(np.array(y_data) == 1), sum(np.array(y_data) == 0)))
    print("Original pos/neg in train {}/{}".format(sum(np.array(y_train) == 1), sum(np.array(y_train) == 0)))
    print("Original pos/neg in test {}/{}".format(sum(np.array(y_test) == 1), sum(np.array(y_test) == 0)))
    X_train, y_train = TomekLinks().fit_resample(X_train, y_train)
    print("After undersampling pos/neg in train {}/{}".format(sum(np.array(y_train) == 1), sum(np.array(y_train) == 0)))
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
    #X_train, y_train = BorderlineSMOTE(random_state=42, kind='borderline-2').fit_resample(X_train, y_train)
    print("After oversampling pos/neg in train {}/{}".format(sum(np.array(y_train) == 1), sum(np.array(y_train) == 0)))
    

    # convert the arrays to torch tensors
    X_train, y_train = torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float).unsqueeze(1)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float).unsqueeze(1)
    print("X_train shape {}".format(X_train.shape))
    print("y_train shape {}".format(y_train.shape))
    return X_train, y_train, X_test, y_test


def train(args, model, device, X_trains, y_trains, optimizer, criterion):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    val_losses = []
    
    # training with 3-fold cross validation
    kf = KFold(n_splits=3, shuffle=True)
    for train_idx, val_idx in kf.split(X_trains):
        #####################################
        # split for training and validation #
        #####################################
        train_dataset = interviewDataset(X_trains[train_idx], y_trains[train_idx])
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
        
        val_dataset = interviewDataset(X_trains[val_idx], y_trains[val_idx])
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=False)
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad()
            X_train, y_train = X_train.to(device), y_train.to(device)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            if batch_idx % args.log_interval == 0:
                print('[{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                       batch_idx * len(X_train), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # log the training loss
            train_losses.append(loss.item())
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            y_trues, y_preds = [], []
            correct = 0
            for batch_idx, (X_val, y_val) in enumerate(val_loader):
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                y_pred = (outputs > 0.5).float()
                correct += (y_pred == y_val).sum().item()
                y_trues.extend(list(y_val.cpu().numpy()))
                y_preds.extend(list(y_pred.cpu().numpy()))
                #print(y_preds)
                
                if batch_idx % args.log_interval == 0:
                    # Print statistics 
                    print("Val Loss: {:.3f}, Accuracy: {:.3f}%".format(loss.item(), 100. * (correct/len(val_loader.dataset))))
                    #cal_confusion_matrix(y_trues, y_preds)
                    
                    # log the val loss
                    val_losses.append(loss.item())
                    
    if args.save_model and avg_val_loss and sum(val_losses)/len(val_losses) < avg_val_loss[-1]:
        print("Save Model ...")
        torch.save(model.state_dict(), "best.pt")
    avg_train_loss.append(sum(train_losses)/len(train_losses))
    avg_val_loss.append(sum(val_losses)/len(val_losses))
        
def test(model, device, test_loader, criterion, epoch):
    ####################    
    #  test the model  #
    ####################
    model.eval()  
    with torch.no_grad():
        # try different cutting threshold for prediction
        thresholds = np.arange(0.1,1,0.1)
        y_trues, y_preds = [], []
        test_losses = []
        for thres in thresholds:
            correct = 0
            _y_preds, _y_trues = [], []
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                outputs = model(X_test)
                loss = criterion(outputs, y_test)
                y_pred = (outputs > thres).float()
                correct += (y_pred == y_test).sum().item()
                _y_preds.extend(list(y_pred.cpu().numpy()))
                _y_trues.extend(list(y_test.cpu().numpy()))
                
                # record the test loss with thres = 0.5 for visualization
                if thres == 0.5:
                    test_losses.append(loss.item())
            # Print statistics 
            print("\nThreshold {:.1f} : Loss: {:.6f}, Accuracy: {:.3f}%".format(thres, loss.item(), 100. * (correct/len(test_loader.dataset))))
            cal_confusion_matrix(_y_trues, _y_preds)
            y_trues.append(_y_trues)
            y_preds.append(_y_preds)
            
            
        # plot confusion matrix with different thresholds
        plot_confusion_matrix(y_trues, y_preds, epoch)
        avg_test_loss.append(sum(test_losses)/len(test_losses))

def cal_confusion_matrix(y_true, y_pred):
    print("----- Confusion Matrix -----")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    df = pd.DataFrame(data=cm, index=['True-Fail', 'True-Pass'], columns=['Pred-Fail', 'Pred-Pass'])
    print(df)
    #print(y_true)
    #print(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    precision = tp/(tp+fp+1e-9)
    recall = tp/(tp+fn+1e-9)
    print("Accuracy : {}".format((tn+tp)/(tn+fp+fn+tp)))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 score : {}".format(2./(1.0/(precision+1e-9)+1.0/(recall+1e-9)+1e-9)))

def plot_confusion_matrix(y_trues, y_preds, epoch):
    # different thresholds for prediction
    thres = np.arange(0.1,1,0.1)
    fig = plt.figure(figsize=(16,16))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for idx, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):
        ax = fig.add_subplot(3, 3, idx+1)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        df_cm = pd.DataFrame(cm, columns=['Fail', 'Pass'], index=['Fail', 'Pass'])
        sns.set(font_scale=1.4)
        sns.heatmap(df_cm, fmt="d", linewidths=3, cmap='PuRd', annot=True, annot_kws={"size": 20})
        ax.set_title("Threshold = {:.1f}".format(thres[idx]))
        ax.set_xlabel("Predict Label")
        ax.set_ylabel("True Label")
    plt.savefig(os.path.join(directory,"ep{}_confusion_mat.jpg".format(str(epoch))))
    plt.show()   
    
def plot_loss_curve(train_loss, val_loss, test_loss):
    plt.figure(figsize=(16,12))
    plt.plot(range(len(train_loss)), train_loss, label='Train Loss')
    plt.plot(range(len(val_loss)), val_loss, label='Val Loss')
    plt.plot(range(len(test_loss)), test_loss, label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(directory, "loss_curve.jpg"))
    plt.show()
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Interview Data Classification')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.005)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='Loading into the current Model')
    parser.add_argument('--model_name', type=str, default='best.pt',
                        help='Model name for loading')
    
    args = parser.parse_args()
    
    # check whether to use gpu or cpu for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # perform data process
    X_train, y_train, X_test, y_test = dataProcess(filename='interviewData.csv', test_percent=0.3)
    
    # Load the test dataset to dataloader
    test_dataset = interviewDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    '''
    # Test data loader
    print("There is {} batches in the dataset".format(len(test_loader)))
    for (x,y) in test_loader:
        print("For one iteration (batch), there is:")
        print("Data:    {}".format(x.shape))
        print("Labels:  {}".format(y.shape))
        break
    '''
    
    model = net(X_train.shape[1]).to(device)
    print(model)
    
    # Check for model loading
    if args.load_model:
        model.load_state_dict(torch.load(args.model_name))
        
    # Binary Cross-entropy
    criterion = torch.nn.BCELoss(reduction='mean')   
    
    # optimizer choices
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning Rate (lr) adjustment
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # initialize the early_stopping object
    #early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    for epoch in range(1, args.epochs + 1):
        print("\n======= Epoch {} ======".format(epoch))
        print("###### Training ######")
        train(args, model, device, X_train, y_train, optimizer, criterion)

        print("###### Testing ######")
        test(model, device, test_loader, criterion, epoch)
        
        # learning rate adjustment
        scheduler.step()
    
    # plot loss curve
    #plot_loss_curve(avg_train_loss, avg_val_loss, avg_test_loss)
    '''
    if args.save_model:
        torch.save(model.state_dict(), "hw.pt")
    '''
if __name__ == '__main__':
    main()
