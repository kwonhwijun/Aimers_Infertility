import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class Preprocessor():
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        # 원본 데이터 기록
        self.raw = self.data.copy()
        return self.data

    def fix_vals(self, df):
        '''
        데이터에서 잘못 입력된 값을 수정하는 함수
            - '배란 유도 유형' 칼럼에서 기록되지 않은 시행은 np.nan으로
            - '난자 출처' 칼럼에서 "알 수 없음" 값은 "본인 제공"으로 변경 (?)
            -   
        '''

        if df is None:
            print("load_data 먼저 하셈")

        if '배란 유도 유형' in df.columns :
            df['배란 유도 유형'] = df['배란 유도 유형'].replace('기록되지 않은 시행', np.nan)

                # '난자 출처': "알 수 없음" → "본인 제공"
        if '난자 출처' in df.columns:
            df['난자 출처'] = df['난자 출처'].replace("알 수 없음", "본인 제공")
        
        # '정자 기증자 나이': "알 수 없음" 여부를 새 변수에 기록
        if '정자 기증자 나이' in df.columns:
            df["정자 기증자 나이_알 수 없음"] = df["정자 기증자 나이"].apply(lambda x: 1 if x == "알 수 없음" else 0)
        
        # '난자 기증자 나이': "알 수 없음" 여부를 새 변수에 기록
        if '난자 기증자 나이' in df.columns:
            df["난자 기증자 나이_알 수 없음"] = df["난자 기증자 나이"].apply(lambda x: 1 if x == "알 수 없음" else 0)
        
        # '배란 유도 유형' 칼럼 삭제 (존재할 경우)
        if '배란 유도 유형' in df.columns:
            df = df.drop(columns=["배란 유도 유형"])

        return df
    
    def impute(self, df):
        '''
        결측 대체 함수 
            - 수치형은 중앙값
            - 범주형은 Unknow 으로 
        '''
        
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        # 수치형 변수 결측치 대체: 중앙값
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # 범주형 변수 결측치 대체: 'Unknown'
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")


        return df
    
    def impute_mice(self, df) :
        '''MICE로 결측 대체 '''
        from sklearn.experimental import enable_iterative_imputer  # sklearn 0.20+ 버전부터
        from sklearn.impute import IterativeImputer

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        return df

    def fit(self):
        self.load_data()
        self.data = self.fix_vals(self.data)
        self.data = self.impute(self.data)

        self.y = self.data["임신 성공 여부"]
        X = self.data.drop("임신 성공 여부", axis=1)
        
        categorical_features = [col for col in X.columns if X[col].dtype == 'object']
        numeric_features = [col for col in X.columns if X[col].dtype != 'object']
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        self.X_processed = self.preprocessor.fit_transform(X)
        
        return self.X_processed, self.y
    
    def transform(self, new_data):
        df = new_data.copy()
        df = self.fix_vals(df)
        df = self.impute(df)
        if '임신 성공 여부' in df.columns :
            df.drop(columns = ['임신 성공 여부'])

        X_transformed = self.preprocessor.transform(df)

        return X_transformed

    def fit_transform(self) :
        return self.fit()

        

    
    


# 잘됐는지 확인 가능 : 터미널에 python3 (경로)/preprocessing.py 입력
if __name__ == '__main__':
    preproc = Preprocessor('/Users/hj/projects/Aimers/data/raw/train.csv')
    X, y = preproc.fit_transform()

    print("전처리 확인")
    print(X.shape)
    print(y.shape)

    test_path = '/Users/hj/projects/Aimers/data/raw/test.csv'
    test = pd.read_csv(test_path)
    X_test = preproc.transform(test)
    print(X_test.shape)