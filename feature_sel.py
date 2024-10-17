# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 19:14:57 2022

@author: lwh
"""
#移除低方差特征
from sklearn.feature_selection import VarianceThreshold
import numpy as np
X = np.array([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
selector = VarianceThreshold()
VarianceThreshold_selector = selector.fit(X)
X_selected = VarianceThreshold_selector.transform(X)
#到底选择了哪些特征？
features_selected = VarianceThreshold_selector.get_support()

from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer() 
data = dataset['data']
label =  dataset['target']
selector = VarianceThreshold(threshold=0.1)
VarianceThreshold_selector = selector.fit(data)
data_selected = VarianceThreshold_selector.transform(data)
features_selected = VarianceThreshold_selector.get_support()

#单变量特征筛选
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest,SelectPercentile
dataset = load_breast_cancer() 
data = dataset['data']
label =  dataset['target']
selector = SelectKBest(k = 10)
SelectKBest_selector = selector.fit(data,label)
data_selected = SelectKBest_selector.transform(data)
scores = SelectKBest_selector.scores_
pvalues = SelectKBest_selector.pvalues_
features_selected = SelectKBest_selector.get_support()

#最好还是区分训练集和验证集的
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
dataset = load_breast_cancer() 
data = dataset['data']
label =  dataset['target']
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.2)
selector = SelectKBest(k = 10, score_func = chi2)
SelectKBest_selector = selector.fit(train_data,train_label)
train_data_selected = SelectKBest_selector.transform(train_data)
test_data_selected = SelectKBest_selector.transform(test_data)
scores = SelectKBest_selector.scores_
pvalues = SelectKBest_selector.pvalues_
features_selected = SelectKBest_selector.get_support()


#通过百分比来选择特征
dataset = load_breast_cancer() 
data = dataset['data']
label =  dataset['target']
selector = SelectPercentile(percentile=30)
SelectPercentile_selector = selector.fit(data,label)
data_selected = SelectPercentile_selector.transform(data)
scores = SelectPercentile_selector.scores_
pvalues = SelectPercentile_selector.pvalues_
features_selected = SelectPercentile_selector.get_support()



#基于惩罚项的特征选择
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.datasets import load_breast_cancer
dataset = load_breast_cancer() #多分类的数据集
data = dataset['data']
label =  dataset['target']
lr = Lasso(alpha=0.1)
#带L1惩罚项的逻辑回归作为基模型的特征选择
model = SelectFromModel(lr)
model.fit(data, label)
data_selected = model.transform(data)
mask = model.get_support()

#基于树模型的特征选择
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
dataset = load_breast_cancer() #多分类的数据集
data = dataset['data']
label =  dataset['target'] #分类数据集
#用全部数据集做特征选择
gbdt = GradientBoostingClassifier()
gbdt.fit(data,label)
#GBDT作为基模型的特征选择
model = SelectFromModel(gbdt, prefit=True,threshold='0.5*mean')
data_selected = model.transform(data)
mask = model.get_support()

#也可以只用训练集做特征选择
train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.2)
gbdt = GradientBoostingClassifier()
gbdt.fit(train_data,train_label)
model = SelectFromModel(gbdt, prefit=True,threshold='1.25*mean')
train_data_selected = model.transform(train_data)
test_data_selected = model.transform(test_data)


#递归特征消除RFE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Feature extraction
dataset = load_breast_cancer() 
data = dataset['data']
label =  dataset['target']
# model = RandomForestClassifier()
model = SVC(kernel = 'linear')
rfe = RFE(model, n_features_to_select=10)
fit = rfe.fit(data, label)
data_selected = rfe.transform(data)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))


#RFE+CV = REFCV（交叉验证）常用方法
#REFCV
# 可以尝试不同的方法来进行特征选择，然后将不同的特征选择组合送入完成的流程，最后找结果最好的那组特征。
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import numpy as np
# Feature extraction
dataset = load_breast_cancer() #多分类的数据集
data  = dataset['data']
label = dataset['target']
model = LogisticRegression()
min_features_to_select = 1
rfecv = RFECV(model, min_features_to_select = 1)
# rfecv = RFECV(model, min_features_to_select = 1, scoring = "recall")
fit = rfecv.fit(data, label)
data_selected = rfecv.transform(data)

print("Optimal number of features : %d" % rfecv.n_features_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (accuracy)")
plt.plot(
    range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
    np.mean(rfecv.grid_scores_,axis=1),
)
plt.show()


#包装法 序列特征选择
from sklearn.naive_bayes import GaussianNB
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
dataset = load_breast_cancer() 
data = dataset['data']
label =  dataset['target']
model = GaussianNB()
# Sequential Forward Selection  
sfs = sfs(model,
           k_features=10, 
           forward=False, #forward=True 叫 SFS  如果False SBS
           floating=False,
           verbose=2,
           scoring='roc_auc',
           cv=5,
           n_jobs=4)

sfs = sfs.fit(data, label)
data_selected = sfs.transform(data)

# np.save(r'D:\核磁与机器学习2\核磁与机器学习/data_selected.npy',data_selected)
# a = np.load(r'D:\核磁与机器学习2\核磁与机器学习/data_selected.npy')

print('\nSequential Backforward Selection:')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)

fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')#Ctrl+左键 查看说明
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()
