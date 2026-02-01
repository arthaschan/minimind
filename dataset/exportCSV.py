#!/usr/bin/env python3
# 把 CMExam 里的test_with_annotations.csv 过滤 'Medical Discipline'] == '口腔医学' 的数据
# 并保存为jsonl 格式，导出为mental.jsonl
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  1 16:24:29 2026

@author: arthas
"""

import pandas as pd

 
# 读取CSV文件
df = pd.read_csv('mental.csv')

print('数据基本信息：')
df.info()

# 查看数据集行数和列数
rows, columns = df.shape

if rows < 100 and columns < 20:
    # 短表数据（行数少于100且列数少于20）查看全量数据信息
    print('数据全部内容信息：')
    print(df.to_csv(sep='\t', na_rep='nan'))
else:
    # 长表数据查看数据前几行信息
    print('数据前几行内容信息：')
    print(df.head().to_csv(sep='\t', na_rep='nan')) 

# 过滤出Medical Discipline列为口腔医学的数据
filtered_df = df[df['Medical Discipline'] == '口腔医学']

# 将结果保存为JSON文件


jsonl_path = 'mental.jsonl'
filtered_df.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)