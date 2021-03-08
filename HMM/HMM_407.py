"""
使用已分词的语料进行训练
使用维特比算法对测试集进行分词
输出分词结果并写入data文件夹下的output.txt

61518407李浩瑞 2021.1.1
"""

import pandas as pd
from numpy import *
from math import log
import os
import sys
import json

"""设置全局变量"""
# 读写文件索引
project_path = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件路径的上一级目录
train_path = project_path+r'\data\RenMinData.txt_utf8'  # 拼接训练路径字符串
test_path = project_path+r'\data\mytest.txt'    # 拼接测试路径字符串
output_path = project_path+r'\output.txt'
# 全局变量
States_index = [0, 1, 2, 3]  # 状态索引
STATES = ['B', 'M', 'E', 'S']  # 单个汉字的四种状态
A = {}
B = {}
Pi = {}
word_set = set()  # 所有显式状态的集合
count_dic = {}
neg_infinity = -61518407e+100  # 负无穷表示log(0)
line_num = 0


def Tag(word):
    """
    获取训练集中每段词语的状态标签
    word:list
    tag: list
    """
    tag = []
    if len(word) == 1:
        tag = ['S']
    elif len(word) == 2:
        tag = ['B', 'E']
    else:
        num = len(word) - 2
        tag.append('B')
        tag.extend(['M'] * num)
        tag.append('E')
    return tag


def train(trainset, word_set, Pi, A, B):
    """
    void
    根据训练数据trainset统计频数，
    将所有训练集内出现过的词存入word_set，
    将概率存入Pi,A,B
    """
    # 初始化字典
    for i in STATES:
        Pi[i] = 0.0
        A[i] = {}
        B[i] = {}
        count_dic[i] = 0
        for j in STATES:
            A[i][j] = 0.0

    # 统计频数
    for line in trainset:
        '''读取训练集'''
        line = line.strip()  # 不strip会读到\n
        global line_num
        line_num += 1
        word_list = []
        for k in range(len(line)):
            if line[k] == ' ':
                continue
            word_list.append(line[k]) 
        word_set = word_set | set(word_list)  # “集合或”，获得训练集所有字的集合
        '''从训练集计算状态转移矩阵'''
        line = line.split(' ')  
        line_state = []
        for i in line:
            line_state.extend(Tag(i))  

        Pi[line_state[0]] += 1  
        for j in range(len(line_state)-1):
            A[line_state[j]][line_state[j+1]] += 1  
        for p in range(len(line_state)):
            count_dic[line_state[p]] += 1  
            for state in STATES:
                if word_list[p] not in B[state]:
                    B[state][word_list[p]] = 0.0  
            B[line_state[p]][word_list[p]] += 1
    #log(概率)
    #对概率0取无穷小neg_infinity
    for i in STATES:
        if Pi[i] == 0:
            Pi[i] = neg_infinity
        else:
            Pi[i] = math.log(Pi[i] / line_num)
        for j in STATES:
            if A[i][j] == 0:
                A[i][j] = neg_infinity
            else:
                A[i][j] = math.log(A[i][j] / count_dic[i])
        for j in B[i]:
            if B[i][j] == 0:
                B[i][j] = neg_infinity
            else:
                B[i][j] = math.log(B[i][j] / count_dic[i])


def Viterbi(obs, Pi, A, B):
    """
    return path[state]:list 存储句子的隐状态
    已知Pi,A,B,显状态sentence
    求测试集最有可能的隐状态集合
    注意概率已经取过log,乘变加
    """
    V = [{}]  # 动态规划表 tabular
    path = {}
    if obs[0] not in B['B']:
        # 防止没见过的隐状态，设置为S，不可能B,M,E,减少传递过程中计算错误的可能
        for i in STATES:
            if i == 'S':
                B[i][obs[0]] = 0
            else:
                B[i][obs[0]] = neg_infinity
    for i in STATES:
        V[0][i] = Pi[i] + B[i][obs[0]] 
        path[i] = [i]
    for i in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state0 in STATES:
            items = []
            for state1 in STATES:
                if obs[i] not in B[state0]: 
                    if obs[i-1] not in B[state0]:
                        prob = V[i - 1][state1] + A[state1][state0]
                    else:
                        prob = V[i - 1][state1] + A[state1][state0]
                else:
                    prob = V[i-1][state1] + A[state1][state0] + \
                        B[state0][obs[i]] 
                items.append((prob, state1))
            best = max(items)
            V[i][state0] = best[0]
            new_path[state0] = path[best[1]] + [state0]
        path = new_path
    prob, state = max([(V[len(obs) - 1][state], state) for state in STATES])
    return path[state]


def test(testset, Pi, A, B):
    """
    void
    对测试数据进行分词
    输出分词结果并写入output.txt
    testset = open(test_path) 
    Pi,A,B:doc
    output:list
    """
    output = ''
    for line in testset:
        line = line.strip()
        tag = Viterbi(line, Pi, A, B)
        splited = []
        start = -1
        started = False
        if len(tag) != len(line):
            return None
        elif len(tag) == 1:
            splited.append(line[0])  # 语句只有一个字，直接输出
        else:
            for i in range(len(tag)):
                if tag[i] == 'S':
                    if started:
                        started = False
                        splited.append(line[start:i])
                    splited.append(line[i])
                elif tag[i] == 'B':
                    if started:
                        splited.append(line[start:i])
                    start = i
                    started = True
                elif tag[i] == 'E':
                    started = False
                    word = line[start:i + 1]
                    splited.append(word)
                elif tag[i] == 'M':
                    continue
        list = ''
        for i in range(len(splited)):
            list = list + splited[i] + ' '
        output = output + list + '\n'
    print("mytestoutput:", output)
    outputfile = open(output_path, mode='w', encoding='utf-8')
    outputfile.write(output)


if __name__ == '__main__':
    """进行训练"""
    trainset = open(train_path, encoding='utf-8')  # 读取训练集
    train(trainset, word_set, Pi, A, B)
    """保存模型"""
    with open(project_path+r"model_Pi.json", "w") as f:
        f.write(json.dumps(Pi, ensure_ascii=True, indent=4, separators=(',', ':')))
    with open(project_path+r"model_A.json", "w") as f:
        f.write(json.dumps(A, ensure_ascii=True, indent=4, separators=(',', ':')))
    with open(project_path+r"model_B.json", "w") as f:
        f.write(json.dumps(B, ensure_ascii=True, indent=4, separators=(',', ':')))
    """进行测试"""
    testset = open(test_path, encoding='utf-8')  # 读取测试集
    test(testset, Pi, A, B)
