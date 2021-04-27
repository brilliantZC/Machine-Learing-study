# -*- coding = utf-8 -*-
# @Time : 2021/4/25 9:52
# @Author : brilliantZC
# @File : sklearntest.py
# @Software : PyCharm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
import jieba
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

def datasets_demo():
    # sklearn数据集的使用

    # 1·获取数据
    iris = load_iris()
    print("鸢尾花数据集：\n",iris)
    print("查看数据集描述：\n",iris["DESCR"])
    print("查看特征值的名字：\n",iris.feature_names)
    print("查看特征值：\n",iris.data,iris.data.shape)

    # 2·数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_test.shape)

    return None

"""
字典特征提取 sklearn.feature_extraction
应用场景：
 1）pclass, sex 数据集当中类别特征比较多
      1、将数据集的特征-》字典类型
      2、DictVectorizer转换
 2）本身拿到的数据就是字典类型
"""
def dict_demo():
    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]

    # 1.实例化一个转换器
    transfer = DictVectorizer(sparse=False)

    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)
    print("特证名字:\n",transfer.get_feature_names())

    return None

# 文本特征提取:CountVectorizer 统计每个样本特征词出现的个数
# stop_words停用的
def count_demo():
    data = ["life is short,i like python","life is too long,i dislike python python"]
    # 1.实例化一个转换器类
    transfer = CountVectorizer(stop_words=["is","too"])

    # 2.调用fit_transfer
    data_new = transfer.fit_transform(data)
    print("特证名字:\n", transfer.get_feature_names())
    # print("data_new:\n", data_new)
    print("data_new:\n", data_new.toarray())

    return None

# 中文文本特征提取
def count_chinese_demo():
    data = ["人生苦短，我喜欢Python", "生活太长久，我不喜欢Python"]
    # 1.实例化一个转换器类
    transfer = CountVectorizer()

    # 2.调用fit_transfer
    data_new = transfer.fit_transform(data)
    print("特证名字:\n", transfer.get_feature_names())
    # print("data_new:\n", data_new)
    print("data_new:\n", data_new.toarray())

    return None

# 进行中文分词
def cut_word(text):
    a = " ".join(list(jieba.cut(text)))
    return a

# 中文文本特征抽取，使用自动分词jieba
def count_chinese_demo2():
    # 1.将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    # 2.实例化一个转换器类
    transfer = CountVectorizer(stop_words=["一种","所以"])

    # 3.调用fit_transfer
    data_final = transfer.fit_transform(data_new)
    print("特证名字:\n", transfer.get_feature_names())
    # print("data_new:\n", data_new)
    print("data_new:\n", data_final.toarray())

    return None

"""
TfidfVectorizer TF-IDF - 重要程度
    两个词 “经济”，“非常”
    1000篇文章-语料库
    100篇文章 - "非常"
    10篇文章 - “经济”
    两篇文章
    文章A(100词) : 10次“经济” TF-IDF:0.2
        tf:10/100 = 0.1
        idf:lg 1000/10 = 2
    文章B(100词) : 10次“非常” TF-IDF:0.1
        tf:10/100 = 0.1
        idf: log 10 1000/100 = 1
    TF - 词频（term frequency，tf)
    IDF - 逆向文档频率
"""

def tfidf_demo():
    # 1.将中文文本进行分词
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))

    # 2.实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=["一种", "所以"])

    # 3.调用fit_transfer
    data_final = transfer.fit_transform(data_new)
    print("特证名字:\n", transfer.get_feature_names())
    print("data_new:\n", data_final.toarray())

    return None

# 归一化
def minmax_demo():
    # 1.获取数据(dating.txt中)
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]  # 第一个：后的数字取行数，第二个：后的数字为取得列数。拿到前三列数据

    # 2.实例化一个转换器
    transfer = MinMaxScaler()

    # 3.调用fit_transfer
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)

    return None

# 标准化(防止异常点的影响)：在已有样本足够多的情况下比较稳定，适合现代嘈杂大数据场景
def stand_demo():
    # 1.获取数据(dating.txt中)
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]  # 第一个：后的数字取行数，第二个：后的数字为取得列数。拿到前三列数据

    # 2.实例化一个转换器
    transfer = StandardScaler()

    # 3.调用fit_transfer
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None

# 过滤低方差特征
def variance_demo():
    # 1.获取数据
    data = pd.read_csv("factor_returns.csv")
    data = data.iloc[:, 1:-2]
    # print(data)
    # 2.实例化一个转换器类
    transfer = VarianceThreshold(threshold=10)

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new,data_new.shape)

    # 计算某两个变量之间的相关性
    """
    特征与特征之间相关性很高：
         1）选取其中一个
         2）加权求和
         3）主成分分析
    """
    r1 = pearsonr(data['pe_ratio'],data['pb_ratio'])
    print("相关系数：\n",r1)
    r2 = pearsonr(data['revenue'],data['total_expense'])
    print("相关系数：\n", r2)
    return None

# PCA降维
def pca_demo():
    data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]

    # 1.实例化转换器类
    transfer = PCA(n_components=2)
    # 2.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new：\n",data_new)

    return None

if __name__ == '__main__':
    # 1.sklearn数据集使用
    # datasets_demo()

    # 2.字典特征提取
    # dict_demo()

    # 3.文本特征提取
    # count_demo()

    # 4.中文文本特征提取
    # count_chinese_demo()

    # 5.中文分词
    # count_chinese_demo2()

    # 6.tfidf
    # tfidf_demo()

    # 7.归一化
    # minmax_demo()

    # 8.标准化
    # stand_demo()

    # 9.低方差特征过滤
    # variance_demo()

    # 10.pca降维
    pca_demo()

