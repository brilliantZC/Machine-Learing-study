# -*- coding = utf-8 -*-
# @Time : 2021/4/26 15:54
# @Author : brilliantZC
# @File : KNN.py
# @Software : PyCharm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# KNN算法对鸢尾花种类预测
def knn_iris():
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state=22)
    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法估计器
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    # 5.模型评估
    # 方法1:直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2:计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率为：\n",score)

    return None

# 用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
def knn_iri_gscv():
    # 1.获取数据
    iris = load_iris()
    # 2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data, iris.target, random_state=22)
    # 3.特征工程：标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4.KNN算法估计器
    estimator = KNeighborsClassifier(n_neighbors=3)

    # 加入网格搜索和交叉验证
    # 参数准备
    param_dict = {"n_neighbors":[1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)

    estimator.fit(x_train,y_train)
    # 5.模型评估
    # 方法1:直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2:计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率为：\n",score)

    print("最佳参数：\n",estimator.best_params_)
    print("最佳结果：\n",estimator.best_score_)
    print("最佳估计器：\n",estimator.best_estimator_)
    print("交叉验证结果：\n",estimator.cv_results_)

    return None


if __name__ == '__main__':

    # knn_iris()
    # 用KNN算法对鸢尾花进行分类，添加网格搜索和交叉验证
    knn_iri_gscv()


"""
优点：简单，易于理解，易于实现，无需训练
缺点：
   1）必须指定K值，K值选择不当则分类精度不能保证
   2）懒惰算法，对测试样本分类时的计算量大，内存开销大
使用场景：小数据场景，几千～几万样本，具体场景具体业务去测试
"""

