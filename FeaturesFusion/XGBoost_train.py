# _*_ encoding:utf-8 _*_
# Author: Lg
# Date: 19/4/30
'''
    使用XGBoost对提取的特征进行强化学习
'''
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.externals import joblib
from sklearn.utils import shuffle
from collections import Counter

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)  # 最大行数
pd.set_option('display.max_columns', 500)  # 最大列数
pd.set_option('display.width', 4000)  # 页面宽度


def loadData():
    data_path = './DEselect_result/101_2.8DE_select.xlsx'
    dataFrame = pd.read_excel(data_path)
    dataFrame = shuffle(dataFrame)
    target = 'zLabel'
    predictors = [x for x in dataFrame.columns if x not in [target]]

    train = dataFrame[predictors]
    label = dataFrame[target]
    train_x, test_x, train_y, test_y = train_test_split(train, label, test_size=0.2)

    return train_x, train_y, test_x, test_y

# 加载数据
train_X, train_y, test_X, test_y = loadData()


# 建立模型
def modelFit(alg, useTrainCV=True, cv_folds=5, eraly_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(train_X, label=train_y)    # 数据使用方式
        cvresult = xgb.cv(xgb_param, xgtrain,    # 使用交叉验证
                          num_boost_round=alg.get_xgb_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='merror',   # 该数据集无法使用AUC进行验证
                          early_stopping_rounds=eraly_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])

    alg.fit(train_X, train_y, eval_metric='auc')

    print('The best paramters is:\n', alg.get_xgb_params())

    dtrain_predictions = alg.predict(train_X)
    dtrain_predprob = alg.predict_proba(train_X)[:, 1]

    print('\n Model Report:\n')
    print('Accuracy(Train): %.4g' %
          metrics.accuracy_score(train_y, dtrain_predictions))
    # print('AUC Score(Train): %f' % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    # feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)   # 对特征重要性进行排序
    # feat_imp.plot(kind='bar', title='Feature Importances')
    # plt.ylabel('Feature Importances Score')
    # plt.grid()
    # plt.show()


# 调参第一步
# max_depth=11：树的最大深度，可以避免过拟合，越深越容易学习到局部的样本特征
# min_child_weight=1：最小叶子结点样本权重和，可以避免过拟合
def step1():
    param_test1 = {'max_depth': range(3, 15, 2),       # 第一次 max_depth=11；min_child_weight=1
                   'min_child_weight': range(1, 10, 2)}
    param_test2 = {'max_depth': [10, 11, 12],            # 第二次 max_depth=1；min_child_weight=7
                   'min_child_weight': [0.5, 1, 1.5]}
    xgb = XGBClassifier(learning_rate=0.1,
                        n_estimators=145,   # 145
                        max_depth=5,
                        min_child_weight=1,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='multi:softmax',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27)
    gsearch = GridSearchCV(estimator=xgb, param_grid=param_test1,
                           scoring='accuracy', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_y)
    print(gsearch.best_params_, gsearch.best_score_)


# 调参第二步
# gamma=0.1：节点分裂所需的最小损失函数下降值（只有分裂后损失函数的下降了才会分类这个节点）
def step2():
    param_test1 = {'gamma': [i/10.0 for i in range(0, 9)]}
    xgb = XGBClassifier(learning_rate=0.1,
                        n_estimators=145,
                        max_depth=11,   # 10
                        min_child_weight=1,   # 7
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='multi:softmax',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27)
    gsearch = GridSearchCV(estimator=xgb, param_grid=param_test1,
                           scoring='accuracy', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_y)
    print(gsearch.best_params_, gsearch.best_score_)


# 调参第三步
# subsample=0.8：控制每棵树随机采样的比例，可以避免过拟合
# colsample_bytree=0.7：控制每棵树随机采样的列数（特征）的比例
def step3():
    param_test1 = {'subsample': [i/10.0 for i in range(6, 11)],        # 第一次 subsample=0.9; colsample_bytree=0.6
                   'colsample_bytree': [i/10.0 for i in range(6, 11)]}
    param_test2 = {'subsample': [i/100.0 for i in range(80, 100, 5)],        # 第二次 subsample=0.9; colsample_bytree=0.55
                   'colsample_bytree': [i/100.0 for i in range(50, 70, 5)]}
    xgb = XGBClassifier(learning_rate=0.1,
                        n_estimators=127,   # 127
                        max_depth=11,
                        min_child_weight=1,
                        gamma=0.1,   # 0.5
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective='multi:softmax',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27)
    gsearch = GridSearchCV(estimator=xgb, param_grid=param_test2,
                           scoring='accuracy', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_y)
    print(gsearch.best_params_, gsearch.best_score_)


# 调参第四步
# reg_alpha=：权重的L1正则化项
# reg_lambda=：权重的L2正则化项
def step4():
    # reg_alpha=0.1  0.9038
    param_test1 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 10, 100]}
    # reg_lambda=0.01 0.9089
    param_test2 = {'reg_lambda': [1e-5, 1e-2, 0.1, 1, 10, 100]}
    xgb = XGBClassifier(learning_rate=0.1,
                        n_estimators=96,   # 96
                        max_depth=10,
                        min_child_weight=7,
                        gamma=0.5,
                        subsample=0.75,   # 0.75
                        colsample_bytree=0.75,   # 0.75
                        objective='multi:softmax',
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27)
    gsearch = GridSearchCV(estimator=xgb, param_grid=param_test1,
                           scoring='accuracy', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_y)
    print(gsearch.best_params_, gsearch.best_score_)


# 在每次网格搜索后使用交叉验证
# 通过将学习率从0.1改为0.01，分类器的个数变为648，训练集上的准确率为0.9912
def doCVafterGridSearch():
    xgb = XGBClassifier(learning_rate=0.01,
                        n_estimators=5000,
                        max_depth=11,
                        min_child_weight=1,
                        gamma=0.1,
                        subsample=0.9,
                        colsample_bytree=0.55,
                        objective='multi:softmax',
                        num_class=3,
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27)
    modelFit(xgb)


if __name__ == '__main__':
    # 调参过程
    step1()
    # doCVafterGridSearch()

    # 验证过程  测试集上的准确率为0.8377
    # grid = 10
    # workSpace_dir = '/Users/manfestain/Workspace/python/WorkSpace/'
    # save_path = workSpace_dir + \
    #     '/MyOperationData/SegModelData/ModelWeights/xgbClassifier_Seg_' + \
    #     str(grid) + '.h5'
    # xgb = XGBClassifier(learning_rate=0.01,
    #                     n_estimators=165,
    #                     max_depth=11,
    #                     min_child_weight=1,
    #                     gamma=0.1,
    #                     subsample=0.9,
    #                     colsample_bytree=0.55,
    #                     objective='multi:softmax',
    #                     num_class=3,
    #                     nthread=4,
    #                     scale_pos_weight=1,
    #                     seed=27)
    # xgb.fit(train_X, train_y)
    # train_result = xgb.predict(train_X)
    # # print(Counter(train_result))
    # print(Counter(train_y))
    # joblib.dump(xgb, save_path)
    # test_predictions = xgb.predict(test_X)
    # print(test_predictions)
    # print(test_y)

    # print('\n Model Report:\n')
    # print('Accuracy(Test): %.4g\n' %
    #       metrics.accuracy_score(test_y, test_predictions))
