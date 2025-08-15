import pandas as pd

class OneHotEncoding:
    def __init__(self, train, test):
        if isinstance(train, pd.DataFrame):
            self.train = train.copy()
        else:
            self.train = pd.read_csv(train)

        if isinstance(test, pd.DataFrame):
            self.test = test.copy()
        else:
            self.test = pd.read_csv(test)
    

    # 필요없는 열 삭제 (ID,이름)
    def PreProcess(self):
        self.test_x = self.test.drop(columns = ['Player_ID', 'Player_Name'])

        self.train_x = self.train[self.test_x.columns].copy()
       
        self.train_y = self.train['Market_Value_Million_EUR'].copy()
    

    # one-hot 인코딩 
    def onehot(self):

        self.one_hot_features = ['Position', 'Preferred_Foot','Nationality', 'Current_Club', 'League']

        self.train_one_hot = pd.get_dummies(self.train_x[self.one_hot_features],dtype = int)
        self.test_one_hot = pd.get_dummies(self.test_x[self.one_hot_features],dtype = int)

        # train의 열 중 test에 없는 열 -> test 열 생성 후 0으로 채움 
        self.test_one_hot = self.test_one_hot.reindex(columns = self.train_one_hot.columns, fill_value = 0)


    # onehot 인코딩 적용 코드     
    def onehot_merge(self):
        self.train_x.drop(self.one_hot_features, axis = 1, inplace = True)
        self.test_x.drop(self.one_hot_features, axis = 1, inplace = True)

        self.train_x = pd.concat([self.train_x,self.train_one_hot], axis = 1)
        self.test_x = pd.concat([self.test_x,self.test_one_hot], axis = 1)


    def run(self):
        self.PreProcess()
        self.onehot()
        self.onehot_merge()
        return self.train_x, self.train_y, self.test_x
    










