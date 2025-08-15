import pandas as pd
import numpy as np

class CareerPreprocessor:
    def __init__(self, train, career_path):
        self.train = train
        self.career_path = career_path

    def process(self):
        # 파일 경로 불러오기 
        career = pd.read_csv(self.career_path)  
        career.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 결측치 -> 중앙값 
        numeric_cols = career.select_dtypes(include=['number']).columns
        career[numeric_cols] = career[numeric_cols].fillna(career[numeric_cols].median())
        
        # train data와 병합 
        career = career[['Player_ID', 'national_caps', 'league_titles', 'ucl_titles', 'season_awards']]
        self.train = pd.merge(self.train, career, how='left', on='Player_ID')
        return self.train