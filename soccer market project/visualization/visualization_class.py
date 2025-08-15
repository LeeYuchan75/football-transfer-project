import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Visualization:
    def __init__(self, train, test):
        self.train = train
        self.test= test 
    
    def process(self):
        self.train.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.test.replace([np.inf, -np.inf], np.nan, inplace=True)

        def wrangling(train_set):  # 데이터셋 정보 확인 
    
            print("---Shape---")
            print(train_set.shape)
    
            print("---Info----")
            print(train_set.info())
    
            print("---NaN----")
            print(train_set.isna().sum())
    
            print("---Duplicated---")
            print(train_set[train_set.duplicated()])
    
            print("---Description---")
            print(train_set.describe())
     
            print("---Unique---")
            print(train_set.nunique())
       
        wrangling(self.train)
  

        import math  # 행렬 크기 계산을 위함 (시각자료 표현)

        num_cols = self.train.select_dtypes(exclude='object').columns.tolist()
        n = len(num_cols)
        cols = 3
        rows = math.ceil(n / cols)

        plt.figure(figsize=(14, rows * 4))

        for i, col in enumerate(num_cols, 1):
            plt.subplot(rows, cols, i)
            sns.histplot(self.train[col])
            plt.title(col)

        plt.tight_layout()
        plt.show()



