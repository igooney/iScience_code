# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:28:41 2022

@author: 80724
"""
import pandas as pd
import numpy as np
from seglearn.transform import FeatureRep, SegmentX,SegmentXY
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,Normalizer ,MinMaxScaler,QuantileTransformer,PowerTransformer
from sklearn.preprocessing import KernelCenterer,RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import scipy.io
import numpy as np
from seglearn.transform import FeatureRep , SegmentX
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassiﬁer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from scipy import stats

#读取.mat文件
mat_file_path = r"D:\核磁与机器学习2\核磁与机器学习\part5-6\fMRI_features\features_AAL116.mat"
mat_file = scipy.io.loadmat(mat_file_path)

Alff_data = mat_file['alff_data']
fAlff_data = mat_file['fAlff_data']
ReHo_data = mat_file['ReHo_data']

features = np.hstack([Alff_data,fAlff_data,ReHo_data])

#读取label
labels = np.loadtxt(r'D:\核磁与机器学习2\核磁与机器学习\part5-6\orginal_fMRI_data\orginal_data\label.txt')
labels = labels - 1


#通过双样本t检验执行特征选择（选择法1）
# positive_features = []
# negative_features = []
# # for i in range(len(features)):
# for i in range(np.size(features,0)):    
#     if labels[i] == 1:
#         positive_features.append(features[i])
#     else:
#         negative_features.append(features[i])
# positive_features = np.vstack(positive_features)
# negative_features = np.vstack(negative_features)
# t_result = stats.ttest_ind(positive_features, negative_features)
# p_value = t_result[1]
# sort_index  = np.argsort(p_value) #将p值排序,从小到大，p值是越小越好
# #筛选出前20%的特征
# number_of_features = int(np.size(features, 1)*0.2) #12720个特征，提取20%
# feature_selected_index = sort_index[:number_of_features]
# #取p=<0.05的特征
# feature_selected_mask = [p_value <= 0.05][0]
# # #提取出新的特征,筛选出前20%的特征
# # new_data = features[:,feature_selected_index]
# #提取出新的特征,取p=<0.05的特征
# feature_selected = features[:,feature_selected_mask]


# ##通过t检验执行验证筛选（选择法2）
# from sklearn.datasets import load_digits
# from sklearn.feature_selection import SelectKBest, chi2,f_classif
# feature_selected = SelectKBest(f_classif, k=20).fit_transform(features, labels)

# #通过sfs（选择法3）
# import matplotlib.pyplot as plt
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
# from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
# import mlxtend
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression()
# sfs = sfs(model, k_features=30, forward=True,floating=False,verbose=2,cv=5,n_jobs=4)
# sfs = sfs.fit(features, labels)
# print('\nSequential Forward Selection (k=auto):')
# print(sfs.k_feature_idx_)
# print('CV Score:')
# print(sfs.k_score_)
# fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
# plt.title('Sequential Forward Selection (Std±Dev)')
# plt.grid()
# plt.show()


#通过RFECV（选择法4）
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn import preprocessing

#数据预处理，通过StandardScaler进行标准化
scaler = preprocessing.StandardScaler().fit(features)
features = scaler.transform(features)
preprocessing.normalize(features, norm='l2')
model = LogisticRegression()
min_features_to_select = 1
model = LogisticRegression()
rfecv = RFECV(model, min_features_to_select = 1, n_jobs = 4, step = 10)
# rfecv = RFECV(model, min_features_to_select = 1, scoring = "recall")
fit = rfecv.fit(features, labels)
data_selected = rfecv.transform(features)

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



#一步法，直接划分训练集验证集和测试集。首先划 分训练集和测试集，在训练集上通过十折交叉调参，接着在测试集上测试。上述过程可重复N次或再次采用N折交叉。
train_acc_his = []
test_acc_his = []
val_acc_his = []
test_sensitivity_his = []
test_specificity_his = []
test_fpr_his = []
test_tpr_his = []
test_auc_his = []
max_cv_train_acc = 0
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
for i in range(100):#循环N次，每次都是随机划分。其实此处也可以改成K折交叉。
    print('---开始第',i,'次循环---')
    # #只经过一步，在原始数据集上划分训练集、验证集和测试集
    train_data,test_data,train_label,test_label = train_test_split(features,labels,test_size=0.3,random_state=i)
        
    #数据预处理，执行归一化
    train_data = preprocessing.normalize(train_data, norm='l2')
    test_data = preprocessing.normalize(test_data, norm='l2')

    #数据预处理，通过StandardScaler进行标准化
    scaler = preprocessing.RobustScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)


    svm = SVC(probability=True) 
    parameters = {
                    'C':np.linspace(0.00001,10,100),
                    'kernel':['rbf','linear'],
                    'gamma' :np.linspace(0.00001,10,100)
                  }    
    #十折交叉调参，此过程会自动划分验证集
    clf = GridSearchCV(svm,parameters,n_jobs=-1,return_train_score=True)
    #训练模型（此时的训练是十折交叉调3参，包括了验证集的划分）
    clf.fit(train_data,train_label)
    para1 = clf.best_params_
    
    print('para1',para1) #得到的参数刚好在你设置的参数范围中间附近，就代表你的参数范围设置是合理的
    #获得cv的所有结果
    cv_results = clf.cv_results_
    #获得验证集的最优结果
    val_acc = clf.best_score_
    best_index = clf.best_index_ #验证集最好时对应的参数index
    #获取训练集的结果
    train_acc = cv_results['mean_train_score'][best_index]
    #获得评估器，测试在测试集上的结果
    best_model_for_cv = clf.best_estimator_
    test_acc = best_model_for_cv.score(test_data,test_label)
    
    
    #计算其他指标
    test_pre = best_model_for_cv.predict(test_data)
    cm = confusion_matrix(test_label,test_pre) #计算混淆矩阵
    tn, fp, fn, tp = cm.ravel() #通过reval获得混淆矩阵的各个值
    #敏感度
    test_sensitivity = tp/(tp+ fn)
    print('test_sensitivity',test_sensitivity)
    test_sensitivity_his.append(test_sensitivity)
    #特异度
    test_specificity =  tn / (fp + tn)
    print('test_specificity',test_specificity)
    test_specificity_his.append(test_specificity)
    
    test_proba = best_model_for_cv.predict_proba(test_data)
    y_pre = test_proba[:,1] #分类为第1类的概率
    fpr, tpr, thersholds = roc_curve(test_label, y_pre) #先算fpr和tpr
    AUC = auc(fpr, tpr) 
    #添加到list里
    train_acc_his.append(train_acc)
    val_acc_his.append(val_acc)
    test_acc_his.append(test_acc)
    test_fpr_his.append(fpr)
    test_tpr_his.append(tpr)
    test_auc_his.append(AUC)
    print('train_acc',train_acc)
    print('val_acc',val_acc)
    print('test_acc',test_acc)
    
    #保存最好train_acc的模型
    if test_acc > max_cv_train_acc:
        max_cv_train_acc = test_acc 
        best_model = best_model_for_cv
        best_model_param = para1
        best_model_test_data = test_data
        best_model_test_label = test_label
        best_weights = best_model_for_cv.coef_
        

# 汇报训练集的结果，训练的结果一般只汇报acc
train_acc_mean = np.mean(train_acc_his)
train_acc_std = np.std(train_acc_his)
train_acc_max = np.max(train_acc_his)
print('train_acc_mean',train_acc_mean)
print('train_acc_std',train_acc_std)
print('train_acc_max',train_acc_max)


#验证集 只汇报acc
val_acc_mean = np.mean(val_acc_his)
val_acc_std = np.std(val_acc_his)
val_acc_max = np.max(val_acc_his)
print('val_acc_mean',val_acc_mean)
print('val_acc_std',val_acc_std)
print('val_acc_max',val_acc_max)

# 需要汇报准确率的均值和标准差
acc_mean = np.mean(test_acc_his)
acc_std = np.std(test_acc_his)
acc_max = np.max(test_acc_his)
print('test_acc_mean',acc_mean)
print('test_acc_std',acc_std)
print('test_acc_max',acc_max)

test_sensitivity_mean = np.mean(test_sensitivity_his)
test_sensitivity_std = np.std(test_sensitivity_his)
test_sensitivity_max = np.max(test_sensitivity_his)
print('test_sensitivity_mean',test_sensitivity_mean)
print('test_sensitivity_std',test_sensitivity_std)
print('test_sensitivity_max',test_sensitivity_max)

test_specificity_mean = np.mean(test_specificity_his)
test_specificity_std = np.std(test_specificity_his)
test_specificity_max = np.max(test_specificity_his)
print('test_specificity_mean',test_specificity_mean)
print('test_specificity_std',test_specificity_std)
print('test_specificity_max',test_specificity_max)


test_auc_mean = np.mean(test_auc_his)
test_auc_std = np.std(test_auc_his)
test_auc_max = np.max(test_auc_his)
print('test_auc_mean',test_auc_mean)
print('test_auc_std',test_auc_std)
print('test_auc_max',test_auc_max)




# #绘制ROC曲线 在最好的结果上面汇报,选择一个指标作为基础，一般选择acc
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
y_label = best_model_test_label
test_proba = best_model.predict_proba(best_model_test_data)
y_pre = test_proba[:,1] #分类为第1类的概率
fpr, tpr, thersholds = roc_curve(y_label, y_pre) #先算fpr和tpr
roc_auc = auc(fpr, tpr) #再算AUC
plt.plot(fpr,tpr,'r--',label='ROC (area = {0:.2f})'.format(roc_auc),lw=2)#绘制ROC曲线
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('max ROC Curve')
plt.legend(loc="lower right")
plt.show()


#绘制平均roc
mean_fpr=np.linspace(0,1,100)
tpr_his_new = []
for i in range(len(test_tpr_his)):
    tpr_his_new.append(np.interp(mean_fpr,test_fpr_his[i],test_tpr_his[i]))
    tpr_his_new[-1][0] = 0
mean_tpr = np.mean(tpr_his_new,0)
mean_tpr[-1]=1.0
roc_auc = auc(mean_fpr, mean_tpr) #再算AUC
plt.plot(fpr,tpr,'r--',label='ROC (area = {0:.2f})'.format(test_auc_mean),lw=2)#绘制ROC曲线
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('mean ROC Curve')
plt.legend(loc="lower right")
plt.show()


#汇报混淆矩阵
from sklearn.metrics import confusion_matrix
y_pred = best_model.predict(best_model_test_data)
cm = confusion_matrix(best_model_test_label,y_pred) #计算混淆矩阵
#绘制混淆矩阵
import seaborn as sns
sns.heatmap(cm,cmap='YlGnBu',fmt='d',annot=True)




per_labels = labels.copy()
#置换检验
per_test_acc_his = []
wights_per_his = []
for i in range(10000):    
    print('---开始第',i,'次循环---')
    # #只经过一步，在原始数据集上划分训练集、验证集和测试集
    np.random.shuffle(per_labels)
    train_data,test_data,train_label,test_label = train_test_split(features,per_labels,test_size=0.3,random_state=i)
    
    #数据预处理，通过RobustScaler进行标准化
    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    # features_scaled = features #不执行标准化

    #数据预处理，执行归一化
    train_data = preprocessing.normalize(train_data, norm='l2')
    test_data = preprocessing.normalize(test_data, norm='l2')
    clf = SVC(probability=True,C = best_model_param['C'],kernel = best_model_param['kernel'], gamma = best_model_param['gamma']) 
    clf.fit(train_data,train_label)
    per_test_acc = clf.score(test_data,test_label)
    per_test_acc_his.append(per_test_acc)
    wights_per_his.append(SVC.coef_)

    
def permutaion_test(test_score_his,acc):
    score_his = test_score_his.copy()
    for i in range(len(score_his)):
        if acc >= score_his[i]:
            score_his[i] = 1
        else:
            score_his[i] = 0
    p = np.sum(score_his)/len(score_his)
    return 1 - p

print(permutaion_test(per_test_acc_his.copy(),acc_mean))



import seaborn as sns 
import matplotlib.pyplot as plt
sns.set_palette("hls") #设置所有图的颜色，使用hls色彩空间
sns.distplot(per_test_acc_his.copy(),color="r",bins=15,kde=False,norm_hist=False) #kde=true，显示拟合曲线
plt.axvline(acc_mean)
plt.title('Permutation Test')
plt.xlabel('difference')
plt.ylabel('distribution')
plt.show()

#In [1]: from sklearn.model_selection import KFold
#导入相关数据
#In [2]: X = ["a", "b", "c", "d", "e", "f"]
#设置分组这里选择分成3份。
#In [3]: kf = KFold(n_splits=3)
#查看分组结果
#In [4]: for train, test in kf.split(X):
#   ...:     print("%s-%s" % (train, test))
