# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:46:22 2020

@author: zhewei
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

def feature_importance(filename='interviewData.csv', test_percent=0.3):  
    # read csv file
    df = pd.read_csv(filename)
    
    # make target
    y_data = [int(i == 1) for i in list(df['hardbin_FT1'])]
    y_data = np.array(y_data, 'float64')
    
    # drop unnecessary features
    df = df.drop(['wafername_WS1','ecid', 'hardbin_FT1'], 1)
    feature_names = df.columns
    X_data = df.to_numpy()
    
    # ------ Visualize the of top 20 ranking features importances ------ #
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    forest.fit(X_data, y_data)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    #print(len(importances[importances >= 0.002]))
    indices = indices[:20]
    #print(indices[:20])
    
    plt.figure(figsize=(16,12))
    plt.title("Top20_Feature_importances")
    plt.bar(range(len(indices)), importances[indices],
            color="r", align="center")
    plt.xticks(range(len(indices)), feature_names[indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.savefig("Top20_Feature_Imp.jpg")
    plt.show()
    
    
    # ---------- Visualize the dist. of top 3 ranking features ---------- #
    neg_index = []
    pos_index = []
    for i, y in enumerate(y_data):
        if y != 1:
            neg_index.append(i)
        else:
            pos_index.append(i)
            
    fig = plt.figure(figsize=(16,16))
    fig.subplots_adjust(hspace=0.3, wspace=0.1)
    for i, v in enumerate(indices[:3]): 
        ax = fig.add_subplot(3, 1, i+1)
        if i == 2:
            sns.distplot(X_data[pos_index, v], norm_hist=True, color='blue', label='Pass') 
            sns.distplot(X_data[neg_index, v], norm_hist=True, color='red', label='Fail')
            #ax.set_ylim(0, 0.006)
        else:
            sns.distplot(X_data[pos_index, v], bins=50, norm_hist=True, color='blue', label='Pass') 
            sns.distplot(X_data[neg_index, v], bins=50, norm_hist=True, color='red', label='Fail')
        ax.legend()
        ax.set_title(feature_names[v])
    fig.suptitle('Top 3 Feature Importance Dist.', fontsize=24)
    plt.savefig("Top3_Feature_Imp_dist.jpg")

if __name__ == '__main__':
    feature_importance()