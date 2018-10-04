# pyml

Python Machine Learning...

This python package is created for implementing AI algorithm by myself.

## requirement

only numpy and pandas


1. model_selection.train_test_split
1. NLP 数据预处理 需要实现的函数有
    1. TextCleaning:
        1. 将一个字符串，变成单词的列表，其中某些字符串会被修改成同义词以减少误差
    1. CountVectorizer
        1. 输入字符串的列表，返回一个矩阵，表明不同的单词出现的次数