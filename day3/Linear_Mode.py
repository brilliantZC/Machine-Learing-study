# -*- coding = utf-8 -*-
# @Time : 2021/5/7 9:58
# @Author : brilliantZC
# @File : Linear_Mode.py
# @Software : PyCharm

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
from sklearn.metrics import mean_squared_error
# 正规方程的优化方法对波士顿房价进行预测
def linear_Regression():
    # 1、获取数据
    boston = load_boston()

    # 2、划分数据集
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)

    # 3、标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)

    # 5、得出模型
    print("正规方程的权重系数为：\n",estimator.coef_)
    print("正规方程的偏置为：\n",estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n",y_predict)
    error = mean_squared_error(y_test,y_predict)
    print("正规方程均方误差为：\n",error)

    return None


# 梯度下降的优化方法对波士顿房价进行预测
def linear_SGDRegressor():
    # 1、获取数据
    boston = load_boston()

    # 2、划分数据集
    x_train,x_test,y_train,y_test = train_test_split(boston.data,boston.target,random_state=22)

    # 3、标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    estimator = SGDRegressor()
    estimator.fit(x_train,y_train)

    # 5、得出模型
    print("梯度下降的权重系数为：\n",estimator.coef_)
    print("梯度下降的偏置为：\n",estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("梯度下降均方误差为：\n", error)

    return None

# 带有L2正则化的线性回归-岭回归方法对波士顿房价进行预测
def linear_Ridge():
    # 1、获取数据
    boston = load_boston()

    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=22)

    # 3、标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、预估器
    estimator = Ridge(alpha=0.7,max_iter=10000)
    estimator.fit(x_train, y_train)

    # 5、得出模型
    print("岭回归的权重系数为：\n", estimator.coef_)
    print("岭回归的偏置为：\n", estimator.intercept_)

    # 6、模型评估
    y_predict = estimator.predict(x_test)
    print("预测房价：\n", y_predict)
    error = mean_squared_error(y_test, y_predict)
    print("岭回归均方误差为：\n", error)

    return None

if __name__ == '__main__':
    # 正规方程的优化方法对波士顿房价进行预测
    linear_Regression()
    # 梯度下降的优化方法对波士顿房价进行预测
    linear_SGDRegressor()
    # 带有L2正则化的线性回归-岭回归方法对波士顿房价进行预测
    linear_Ridge()
