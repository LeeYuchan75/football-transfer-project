import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("C:/Users/82103/Desktop/축구 이적시장 프로젝트/soccer_data.csv")

train.replace([np.inf, -np.inf], np.nan, inplace=True)

def wrangling(train_set):  # 데이터셋 정보 확인 
    
    print("---Shape---")
    display(train_set.shape)
    
    print("---Info----")
    display(train_set.info())
    
    print("---NaN----")
    display(train_set.isna().sum())
    
    print("---Duplicated---")
    display(train_set[train_set.duplicated()])
    
    print("---Description---")
    display(train_set.describe())
     
    print("---Unique---")
    display(train_set.nunique())
    
wrangling(train)


import math       # 행렬 크기 계산을 위함 (시각자료 표현)

num_cols = train.select_dtypes(exclude='object').columns.tolist()
n = len(num_cols)
cols = 3
rows = math.ceil(n / cols)

plt.figure(figsize=(14, rows * 4))

for i, col in enumerate(num_cols, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(train[col])
    plt.title(col)

plt.tight_layout()



