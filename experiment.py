
from source.preprocessing import Preprocessor
from source.model import Modeler
from source.model_2way import Modeler2way
from source.evaluate import Evaluator
from source.submit import Submisson
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from datetime import datetime
import argparse

from sklearn.model_selection import StratifiedShuffleSplit



# 파일 경로 

train_path = 'data/raw/train.csv'
test_path = 'data/raw/test.csv'

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_name',
        nars = '?',
        default= None,
        help="Specify model filename (without extension) or 'new' for timestamped new model."
    )
    args = parser.parse_args()

    if args.model_name is None:
        # 인수가 없으면 기본 모델 파일 사용
        model_file = "models/catboost_no_one_hot.cbm"
    elif args.model_name.lower() == "new":
        # 'new'라는 인수를 받으면 현재 시간으로 파일명 생성
        today = datetime.now().strftime('%m%d_%H%M%S')
        model_file = f"models/catboost_{today}.cbm"
    else:
        # 사용자가 입력한 모델명으로 파일명 결정 (확장자 .cbm 추가)
        model_file = f"models/{args.model_name}.cbm"

    print(f"[INFO] Model file will be: {model_file}")

    return model_file

def main():
    # 0. 터미널에서 모델명 가져오기 (중요 X, 그냥 기능 용도)
    model_file = parsing()
    
    # 1. 데이터 불러오기
    data = pd.read_csv(train_path)
    y = data['임신 성공 여부']
    X = data.drop(columns = ['임신 성공 여부'])

    X_ivf = X[X['시술 유형'] == 'IVF']
    X_di =  X[X['시술 유형'] == 'DI']
    y_ivf = y[X_ivf.index]
    y_di = y[X_di.index]

    # 2. 데이터 분리 (Stratified 해서 시술 유형 불균형 해결 )

    data_list = {'ivf' : [X_ivf, y_ivf], 'di': [X_di, y_di]}
    for key, value in data_list.items() :
        X, y = value[0], value[1]
        print(f'{key}')
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=73, shuffle= True, stratify=X['시술 유형'])
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 73, shuffle= True, stratify=X_temp['시술 유형'])
        
        # 3. 데이터 전처리 
        preproc =  Preprocessor()
        preproc.fit(X_train)
        X_train_processed = preproc.fit_transform(X_train)
        X_valid_processed = preproc.transform(X_valid)
        X_test_processed = preproc.transform(X_test)

        # 4. 모델 학습 (기존 모델 사용 / 모델 새로 학습 )
        modeler = Modeler()

        print("모델 새로 학습 중")
        print(X_train_processed.shape)
        print(type(X_train_processed))
        modeler.train(X_train_processed, y_train, X_valid_processed, y_valid)
        modeler.model.save_model(model_file)
        
        # 4. 모델 평가
        evaluator = Evaluator()
        print("[Validation Set]")
        evaluator.evaluate(modeler, X_valid_processed, y_valid)

        print("[Test Set]")
        evaluator.evaluate(modeler, X_test_processed, y_test)

    # 5. 제출본 만들기
    from datetime import datetime
    submission = Submisson(preproc)
    today = datetime.now().strftime('%m%d_%H%M%S')
    submission_file = f'submission/submission_{today}.csv'
    submission.gen_submit(modeler, test_file_path= test_path, output_file= submission_file)

    print(f"[INFO] Model file : {model_file}")
    print(f"[INFO] Submission file : {submission_file}")

if __name__ == '__main__':
    main()