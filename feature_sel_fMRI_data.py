# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:14:20 2022

@author: lwh
"""
import mlxtend

#加载特征
import nibabel
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io


#读取.mat文件
mat_file_path = r"D:\核磁与机器学习2\核磁与机器学习\part5-6\fMRI_features\features_AAL116.mat"
mat_file = scipy.io.loadmat(mat_file_path)

Alff_data = mat_file['alff_data']
fAlff_data = mat_file['fAlff_data']
ReHo_data = mat_file['ReHo_data']

features = np.hstack([Alff_data,fAlff_data,ReHo_data])

#读取label
label = np.loadtxt(r'D:\核磁与机器学习2\核磁与机器学习\part5-6\orginal_fMRI_data\orginal_data\label.txt')
label = label - 1















#特征筛选 基于卡方
# from sklearn.feature_selection import SelectKBest, chi2
# chi2feature_select = SelectKBest(chi2, k=20)
# chi2feature_select.fit(data, label)
# p_value = chi2feature_select.pvalues_
# chi2feature_select.get_support()
# new_data = chi2feature_select.transform(data)

# #特征筛选 基于树模型
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import GradientBoostingClassifier
# gbdt = GradientBoostingClassifier()
# gbdt.fit(data,label)
# #GBDT作为基模型的特征选择
# model = SelectFromModel(gbdt, prefit=True)
# new_data = model.transform(data)


#sfs
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import mlxtend
from sklearn.svm import SVC
# x = np.delete(x,[18,19,21,26],1)

# parameters = {
#                 'C':np.linspace(0.00001,10,10),
#                 # 'kernel':range(2,20,1),
#                 'gamma' :np.linspace(0.00001,10,10)
#               }

# RF = SVC()
# RF_set = GridSearchCV(RF, parameters)

svc = SVC()
sfs = sfs(svc,k_features='best', forward=False,floating=False,verbose=2,cv=0,n_jobs=-1)
sfs = sfs.fit(data, label)
print('\nSequential Forward Selection (k=auto):')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (Std±Dev)')
plt.grid()
plt.show()





#SVM的调用方法Python调用：sklearn.svm.SVC，自动调参
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

#整体划分 
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.2,random_state=42)

#实例化
svm = SVC() #C在0-10之间 gamam在0到1之间
#十折交叉调参，此过程会自动划分验证集
parameters = {'kernel':('linear', 'rbf'), 'C':np.linspace(1e-5,10,10),'gamma':np.linspace(1e-4,1,10)}
# clf = GridSearchCV(svm,parameters)
clf = RandomizedSearchCV(svm,parameters,n_jobs=-1)

#训练模型（此时的训练是十折交叉调参，包括了验证集的划分）
clf.fit(train_data,train_label)

#查看最优的结果（验证集）和参数
cv_result = clf.cv_results_
clf.best_score_
para = clf.best_params_
BEST_estimator = clf.best_estimator_
BEST_estimator.score(test_data,test_label) #通过得到的最好的分类器去测试，但其不能充分利用数据集

#用筛选好的参数重新在所有的训练集上去训练模型，充分利用数据
svm = SVC(C=para['C'],kernel=para['kernel'],gamma=para['gamma'])
svm.fit(train_data,train_label)
#测试模型
test_acc = svm.score(test_data,test_label)
svm.predict(test_data)
print(test_acc)





#实例化 包括预处理
#特征预处理
from sklearn import preprocessing
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

#整体划分 
# train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.2,random_state=42)

#标准化
# scaler = preprocessing.StandardScaler().fit(train_data)
# train_data_scaled = scaler.transform(train_data)
# test_data_scaled = scaler.transform(test_data)


#尝试分段标准化
mat_file_path = r"C:\Users\lwh\Desktop\核磁与机器学习\fMRI_features\features_AAL116.mat"
mat_file = scipy.io.loadmat(mat_file_path)

Alff_data = mat_file['alff_data']
fAlff_data = mat_file['fAlff_data']
ReHo_data = mat_file['ReHo_data']

scaler = preprocessing.StandardScaler().fit(Alff_data)
Alff_data_scaled = scaler.transform(Alff_data)

scaler = preprocessing.StandardScaler().fit(fAlff_data)
fAlff_data_scaled = scaler.transform(fAlff_data)

scaler = preprocessing.StandardScaler().fit(ReHo_data)
ReHo_data_scaled = scaler.transform(ReHo_data)

scaled_data = np.hstack([Alff_data_scaled,fAlff_data_scaled,ReHo_data_scaled])

label = np.loadtxt(r'C:\Users\lwh\Desktop\核磁与机器学习\orginal_fMRI_data\orginal_data\label.txt')

train_data_scaled,test_data_scaled,train_label,test_label = train_test_split(data,label,test_size=0.2,random_state=42)

#尝试分段标准化结束


svm = SVC() #C在0-10之间 gamam在0到1之间
#十折交叉调参，此过程会自动划分验证集
parameters = {'kernel':('linear', 'rbf'), 'C':np.linspace(1e-5,10,10),'gamma':np.linspace(1e-4,1,10)}
# clf = GridSearchCV(svm,parameters)
clf = RandomizedSearchCV(svm,parameters,n_jobs=-1)

#训练模型（此时的训练是十折交叉调参，包括了验证集的划分）
clf.fit(train_data_scaled,train_label)

#查看最优的结果（验证集）和参数
cv_result = clf.cv_results_
clf.best_score_
para = clf.best_params_
BEST_estimator = clf.best_estimator_
BEST_estimator.score(test_data_scaled,test_label) #通过得到的最好的分类器去测试，但其不能充分利用数据集

#用筛选好的参数重新在所有的训练集上去训练模型，充分利用数据
svm = SVC(C=para['C'],kernel=para['kernel'],gamma=para['gamma'])
svm.fit(train_data,train_label)
#测试模型
test_acc = svm.score(test_data_scaled,test_label)
print(test_acc)
#查看模型权重
# svm.coef_ 


#换模型
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import numpy as np 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#数据集划分
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.2,random_state=42)
#实例化
DT = DecisionTreeClassifier() 
#十折交叉调参，此过程会自动划分验证集
parameters = {'max_depth':range(2,5,1), 'min_samples_split':range(2,10,1)}
clf = GridSearchCV(DT,parameters)
#训练模型（此时的训练是十折交叉调参，包括了验证集的划分）
clf.fit(train_data,train_label)
para1 = clf.best_params_
#分段调参
DT = DecisionTreeClassifier(max_depth=para1['max_depth'],min_samples_split=para1['min_samples_split']) 
parameters = {'min_samples_leaf':range(2,10,1)}
clf = GridSearchCV(DT,parameters)    
clf.fit(train_data,train_label)
para2 = clf.best_params_  

import numpy as np
his_acc_his = []
max_acc = 0
for i in range(1):
    print(i)
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.1)
    #用筛选好的参数重新在所有的训练集上去训练模型，充分利用数据
    DT = DecisionTreeClassifier(max_depth=para1['max_depth'],min_samples_split=para1['min_samples_split'],
                                 min_samples_leaf=para2['min_samples_leaf'])
    DT.fit(train_data,train_label)
    #测试模型
    test_acc = DT.score(test_data,test_label)
    if test_acc > max_acc:
        max_acc = test_acc
        saved_model = DT
        train_data_saved = train_data
        test_data_saved = test_data
        train_label_saved = train_label
        test_label_saved = test_label
    his_acc_his.append(test_acc)

#结果汇报（均值和标准、最大值）
acc_std = np.std(his_acc_his)
acc_mean = np.mean(his_acc_his)
acc_max = np.max(his_acc_his)



#随机森林
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
import numpy as np 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#数据集划分
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.1,random_state=42)
#实例化
RF = RandomForestClassifier() 
#十折交叉调参，此过程会自动划分验证集
parameters = {'max_depth':range(2,5,1), 'min_samples_split':range(2,10,1)}
clf = GridSearchCV(RF,parameters)
#训练模型（此时的训练是十折交叉调参，包括了验证集的划分）
clf.fit(train_data,train_label)
para1 = clf.best_params_
#分段调参
RF = RandomForestClassifier(max_depth=para1['max_depth'],min_samples_split=para1['min_samples_split']) 
parameters = {'min_samples_leaf':range(2,10,1),'n_estimators':range(10,100,10)}
clf = GridSearchCV(RF,parameters)    
clf.fit(train_data,train_label)
para2 = clf.best_params_  
clf.best_score_


import numpy as np
his_acc_his = []
max_acc = 0
for i in range(1):
    print(i)
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.1)
    #用筛选好的参数重新在所有的训练集上去训练模型，充分利用数据
    RF = RandomForestClassifier(max_depth=para1['max_depth'],min_samples_split=para1['min_samples_split'],
                                 min_samples_leaf=para2['min_samples_leaf'],n_estimators=para2['n_estimators'])
    RF.fit(train_data,train_label)
    #测试模型
    test_acc = RF.score(test_data,test_label)
    if test_acc > max_acc:
        max_acc = test_acc
        saved_model = RF
        train_data_saved = train_data
        test_data_saved = test_data
        train_label_saved = train_label
        test_label_saved = test_label
    his_acc_his.append(test_acc)

#结果汇报（均值和标准、最大值）
acc_std = np.std(his_acc_his)
acc_mean = np.mean(his_acc_his)
acc_max = np.max(his_acc_his)