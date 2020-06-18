# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:31:45 2020

@author: zhewei

--Purpose: perform dimension reduction and visualize the results
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# create directory for storing figures
directory = 'figures'
if not os.path.exists(directory):
    print("Create dir: {}".format(directory))
    os.makedirs(directory) 
    
def dataProcess(filename='interviewData.csv', test_percent=0.2):  
    # read csv file
    df = pd.read_csv(filename)
    
    y_data = [int(i == 1) for i in list(df['hardbin_FT1'])]
    # drop unnecessary features
    df = df.drop(['totaldietestseconds_WS1','wafername_WS1','ecid', 'hardbin_FT1'], 1)
    X_data = df.to_numpy()
    print("Num. of pos/neg data in train {}/{}".format(sum(np.array(y_data) == 1), sum(np.array(y_data) == 0)))
    #X_data, y_data = SMOTE(random_state=42).fit_resample(X_data, y_data)
    #print("len of pos/neg in train {}/{}".format(sum(np.array(y_data) == 1), sum(np.array(y_data) == 0)))
    # Feature Normalization.
    # All features should have the same range of values (-1,1)
    #sc = StandardScaler()
    #X_data = sc.fit_transform(X_data_np)
    
    return X_data, y_data

X_data, y_data = dataProcess()
df = pd.read_csv('interviewData.csv')
neg_index = []
pos_index = []
for i, y in enumerate(y_data):
    if y != 1:
        neg_index.append(i)
    else:
        pos_index.append(i)
        
# ---------- Visualize the dist. with PCA ---------- #
print("======== plot PCA ========")
pca_transformer = PCA(n_components=2)
pca_results = pca_transformer.fit_transform(X_data)
df_results = pd.DataFrame(data = pca_results, columns = ['col_1', 'col_2'])    

plt.figure(figsize=(10,10))
plt.scatter(df_results.loc[pos_index, 'col_1']
           ,df_results.loc[pos_index, 'col_2'], label='Passing', c = 'blue', s = 50)
plt.scatter(df_results.loc[neg_index, 'col_1']
           ,df_results.loc[neg_index, 'col_2'], label='Failing', c = 'red', s = 50)
plt.grid(color='black', linestyle='-', linewidth=1)
plt.legend(fontsize='large')
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component 1',fontsize=20)
plt.ylabel('Principal Component 2',fontsize=20)
plt.title("PCA of Dataset",fontsize=20)
plt.savefig(os.path.join(directory, 'PCA_result.jpg'), dpi=250)
plt.show()

# ---------- Visualize the dist. with tSNE ---------- #
print("======== plot tSNE ========")
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=400)
tsne_results = tsne.fit_transform(X_data)
df_results = pd.DataFrame(data = tsne_results, columns = ['col_1', 'col_2']) 

plt.figure(figsize=(10,10))
plt.scatter(df_results.loc[pos_index, 'col_1']
           ,df_results.loc[pos_index, 'col_2'], label='Passing', c = 'blue', s = 50)
plt.scatter(df_results.loc[neg_index, 'col_1']
           ,df_results.loc[neg_index, 'col_2'], label='Failing', c = 'red', s = 50)
plt.grid(color='black', linestyle='-', linewidth=1)
plt.legend(fontsize='large')
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('tSNE-2d one',fontsize=20)
plt.ylabel('tSNE-2d two',fontsize=20)
plt.title("tSNE of Dataset",fontsize=20)
plt.savefig(os.path.join(directory, 'tSNE_result.jpg'), dpi=250)
plt.show()