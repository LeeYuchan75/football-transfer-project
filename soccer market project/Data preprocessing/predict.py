import numpy as np
from xgboost import XGBRegressor
import optuna
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import KFold # valid data -> KFold (3-fold) 사용
from lightgbm import LGBMRegressor


class Predict_Xgb:
    
    def __init__ (self,train_x,train_y,test_x):
        self.train_x = train_x.copy()
        self.train_y = train_y.copy()
        self.test_x = test_x.copy()

    def XGBOOST(self):

        def objective(trial):

        # 하이퍼파라미터 튜닝
            params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1),  # 학습률 설정
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),  # 최소 자식 가중치 설정
            'gamma': trial.suggest_float('gamma', 0.01, 1.0),  # 감마(리프 노드의 최소 손실 감소) 설정
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),  # L1 정규화 파라미터 설정
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),  # L2 정규화 파라미터 설정
            'seed': 42,  # 랜덤 시드 설정
            'max_depth': trial.suggest_int('max_depth', 3, 15),  # 트리의 최대 깊이 설정
            'n_estimators': trial.suggest_int('n_estimators', 300, 3000, 200),  # 트리의 개수 설정
            'eta': trial.suggest_float('eta', 0.007, 0.013),  # eta(learning_rate의 다른 이름) 설정
            'subsample': trial.suggest_discrete_uniform('subsample', 0.3, 1, 0.1),  # 트레이닝에 사용할 데이터의 비율 설정
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.4, 0.9, 0.1),  # 각 트리의 feature 샘플링 비율 설정
            'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.4, 0.9, 0.1),  # 각 레벨의 feature 샘플링 비율 설정
            }

            # valid data -> kfold 사용
            kf = KFold(n_splits = 3, shuffle = True, random_state = 42)
            fold_scores = []
            
            # K-fold 적용 
            for tr_idx, val_idx in kf.split(self.train_x):
                X_tr = self.train_x.iloc[tr_idx]
                X_val = self.train_x.iloc[val_idx]
                y_tr = self.train_y.iloc[tr_idx]
                y_val = self.train_y.iloc[val_idx]


                # XGBRegressor 객체 생성
                model = XGBRegressor(**params, random_state=42, n_jobs=-1, objective='reg:squaredlogerror', eval_metric='rmsle', early_stopping_rounds=100)

                # XGBoost 모델 훈련 
                bst_xgb = model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

                # 모든 fold의 평균 산출  
                preds = bst_xgb.predict(X_val)
                preds = np.where(preds > 0, preds, 0) 
                rmsle = np.sqrt(msle(y_val, preds))
                fold_scores.append(rmsle)

            return float(np.mean(fold_scores))  # RMSLE (Root Mean Squared Log Error) 반환


        # 하이퍼파라미터 최적화
        study_xgb = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=100))
        study_xgb.optimize(objective, n_trials=1, show_progress_bar=True)  # 1번의 시도에서 최적의 파라미터를 찾습니다.

        # 최종 XGBRegressor 모델 학습
        xgb_reg = XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1, objective='reg:squaredlogerror')
        xgb_reg.fit(self.train_x, self.train_y,verbose=100)

        return xgb_reg, study_xgb  # 최종 모델과 스터디 결과 반환
    

    def LGBM(self):
        
        def objective(trial):
            params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024, step=1, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0, log=False),
            }

            # 3-Fold KFold (XGB와 동일 설정)
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            fold_scores = []

            for tr_idx, val_idx in kf.split(self.train_x):
                X_tr = self.train_x.iloc[tr_idx]
                X_val = self.train_x.iloc[val_idx]
                y_tr = self.train_y.iloc[tr_idx]
                y_val = self.train_y.iloc[val_idx]

                model = LGBMRegressor(**params, random_state=42, n_jobs=-1)
                # 조기종료: LightGBM은 fit의 kwargs로 전달
                bst_lgbm = model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[])

                preds = bst_lgbm.predict(X_val)
                preds = np.where(preds > 0, preds, 0)
                rmsle = np.sqrt(msle(y_val, preds))
                fold_scores.append(rmsle)

            return float(np.mean(fold_scores))

        # Optuna 최적화 
        study_lgbm = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=100))
        study_lgbm.optimize(objective, n_trials=1, show_progress_bar=True)

        # 최적 파라미터로 전체 학습
        lgbm_reg = LGBMRegressor(**study_lgbm.best_params, random_state=42, n_jobs=-1)
        lgbm_reg.fit(self.train_x, self.train_y, eval_metric='rmse')

        return lgbm_reg, study_lgbm
    

    def XGB_LGB_PREDICT(self, xgb_weight=0.1, lgb_weight=0.9, sample_path='sample_submission.csv', output_path='final_submission.csv'):
        
        # 항상 로그 변환 후 학습
        self.train_y = np.log1p(self.train_y)

        # 모델 학습
        xgb_model, _ = self.XGBOOST()
        lgbm_model, _ = self.LGBM()
        
        # (로그 변환 -> 원상태) 복원 후 예측 
        xgb_pred = np.expm1(xgb_model.predict(self.test_x))
        lgbm_pred = np.expm1(lgbm_model.predict(self.test_x))


        # 음수 제거 (RMSLE 대비)
        xgb_pred = np.where(xgb_pred > 0, xgb_pred, 0)
        lgbm_pred = np.where(lgbm_pred > 0, lgbm_pred, 0)

        # 가중 평균
        final_pred = xgb_pred * xgb_weight + lgbm_pred * lgb_weight
    
        print("예측 결과",final_pred)


        
    



        

    

