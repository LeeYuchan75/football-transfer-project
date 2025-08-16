## 🔧 Data Preprocessing (Detailed)

This folder contains scripts for cleaning, transforming, and feature engineering of the dataset.  
Each file has a specific role in preparing the data for modeling.

---

### 📄 Files

- **contract_preprocess.py**  
  - Extracts contract start/end dates into year, month, and day  
  - Handles missing values:  
    - Numerical → median  
    - Categorical (≤5 classes) → mode  
    - Categorical (>5 classes) → `"Unknown"`  

<br/>

- **career_class.py**  
  - Loads career dataset and merges with train data  
  - Adds `national_caps`, `league_titles`, `ucl_titles`, `season_awards`  
  - Fills missing numerical values with median  

<br/>

- **injury_class.py**  
  - Processes injury dataset (date split into year/month/day)  
  - Handles missing values (median/mode/`Unknown`)  
  - Creates a new feature `injury_case_dangerous` representing the impact of injuries  

<br/>

- **onehot_encoder.py**  
  - Performs one-hot encoding on categorical variables  
  - Ensures train/test sets have consistent feature columns
 
<br/>

- **predict.py**  
  - Implements model training with **XGBoost** and **LightGBM**  
  - Hyperparameter tuning with Optuna  
  - Combines predictions using weighted averaging

<br/>

- **data_preprocess.ipynb**  
  - Jupyter Notebook demonstrating the full preprocessing pipeline step by step  

<br/>

---


### 📌 Summary
- Extracted date features and handled missing values consistently  
- Engineered new features from career and injury datasets  
- Applied one-hot encoding for categorical variables  
- Built baseline models with boosting algorithms (XGBoost, LightGBM)
