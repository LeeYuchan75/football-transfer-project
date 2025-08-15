import pandas as pd
import numpy as np

class InjuryPreprocessor:
    def __init__(self, train, injury_path):
        self.train = train
        self.injury_path = injury_path

    def process(self):
        # 파일 경로 불러오기
        injury_df = pd.read_csv(self.injury_path)
        injury_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 날짜 추출 패턴 
        time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2})'
        injury_df[['Injury_year','Injury_month','Injury_day']] = injury_df['Injury_Date'].str.extract(time_pattern).astype('Int64')
        injury_df.drop(columns=['Injury_Date'], inplace=True)
        
        # 수치형 데이터 전처리 
        numeric_cols = injury_df.select_dtypes(include=['number']).columns
        injury_df[numeric_cols] = injury_df[numeric_cols].fillna(injury_df[numeric_cols].median())
        for col in injury_df.select_dtypes(include='object').columns:
            if injury_df[col].isna().sum() > 0:
                if injury_df[col].nunique() <= 5:
                    injury_df[col] = injury_df[col].fillna(injury_df[col].mode()[0])
                else:
                    injury_df[col] = injury_df[col].fillna('unKnown')
        
        # (부상,train) 병합 
        temp_1 = pd.merge(injury_df, self.train[['Player_ID', 'Market_Value_Million_EUR']], how='left', on='Player_ID')
        injury_type_avg_value = temp_1[['Injury_Type','Market_Value_Million_EUR']].groupby('Injury_Type').mean()
        injury_type_avg_value.columns = ['injury_type_avg_value']
        

        temp_1['multiply'] = temp_1['Injury_Duration'] * temp_1['Games_Missed']
        sum_multiply = temp_1[['Injury_Type','multiply']].groupby('Injury_Type').sum().reset_index()
        temp_1.drop(['multiply'],axis = 1, inplace = True)
        
        game_missed_sum = temp_1[['Injury_Type','Games_Missed']].groupby(['Injury_Type']).sum()
        game_missed_sum.columns = ['game_missed_sum']

        temp_2 = pd.merge(temp_1, injury_type_avg_value, how='left', on='Injury_Type')
        temp_2 = pd.merge(temp_2, sum_multiply, how='left', on='Injury_Type')

        temp_2 = pd.merge(temp_2, game_missed_sum, how='left', on='Injury_Type')
        temp_2['injury_case_dangerous'] = temp_2['multiply'] / temp_2['game_missed_sum']

        result = temp_2[['Player_ID', 'injury_case_dangerous']]
        self.train = pd.merge(self.train, result, how='left', on='Player_ID')
        return self.train
        
        
        