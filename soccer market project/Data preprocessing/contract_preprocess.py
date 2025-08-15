import pandas as pd
import numpy as np

class ContractDateProcessor:
    def __init__(self, train):
        self.train = train

    def process(self):
        
        # 날짜 추출 코드 
        time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
        self.train[['Contract_start_year','Contract_start_month','Contract_start_day']] = self.train['Contract_Start'].str.extract(time_pattern).apply(pd.to_numeric)
        self.train[['Contract_end_year','Contract_end_month','Contract_end_day']] = self.train['Contract_End'].str.extract(time_pattern).apply(pd.to_numeric)
        self.train.drop(columns=['Contract_Start', 'Contract_End'], inplace=True)
        
        # 결측치 처리 
        self.train['Player_Name'].replace('', np.nan, inplace=True)
        numeric_cols = self.train.select_dtypes(include=['number']).columns
        self.train[numeric_cols] = self.train[numeric_cols].fillna(self.train[numeric_cols].median())
        
        
        # 결측치 처리 con't : 기준점을 5로 설정하여 결측치 처리
        for col in self.train.select_dtypes(include='object').columns:
            if self.train[col].isna().sum() > 0:
                if self.train[col].nunique() <= 5:
                    self.train[col] = self.train[col].fillna(self.train[col].mode()[0])
                else:
                    self.train[col] = self.train[col].fillna('unKnown')

        return self.train