# -*- coding = utf-8 -*-
# @Time : 2021/5/6 16:25
# @Author : brilliantZC
# @File : DecisionTree.py
# @Software : PyCharm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz


# 用决策树算法对鸢尾花进行分类
def decision_iris():
    # 1、获取数据
    iris = load_iris()

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3、决策树预估器分类
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train,y_train)

    # 4、模型评估
    # 方法1:直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2:计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    # 可视化决策树
    export_graphviz(estimator,out_file="iris_tree.dot",feature_names=iris.feature_names)

    return None


if __name__ == '__main__':
    decision_iris()

"""
优点：
    可视化 - 可解释能力强
缺点：
    容易产生过拟合
"""