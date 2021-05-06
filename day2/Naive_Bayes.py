# -*- coding = utf-8 -*-
# @Time : 2021/5/6 15:41
# @Author : brilliantZC
# @File : Naive_Bayes.py
# @Software : PyCharm

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# 用朴素贝叶斯对新闻进行分类
def nb_news():
    # 1、获取数据
    news = fetch_20newsgroups(subset="all")

    # 2、划分数据集
    x_train,x_test,y_train,y_test = train_test_split(news.data,news.target,random_state=12)

    # 3、特征工程：文本特征抽取-tfidf
    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 4、朴素贝叶斯算法预估器流程
    estimator = MultinomialNB()
    estimator.fit(x_train,y_train)

    # 5、模型评估
    # 方法1:直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("直接比对真实值和预测值:\n", y_test == y_predict)

    # 方法2:计算准确率
    score = estimator.score(x_test, y_test)
    print("准确率为：\n", score)

    return None

if __name__ == '__main__':
    nb_news()
"""
优点：
    对缺失数据不太敏感，算法也比较简单，常用于文本分类。
    分类准确度高，速度快
缺点：
    由于使用了样本属性独立性的假设，所以如果特征属性有关联时其效果不好
"""
