import pandas as pd
import os
import pickle

from source.preprocessing import Preprocessor, split_dataset
from source.model import Modeler, CatBoostPipeline
from source.evaluate import Evaluator
from source.submit import Submisson
from sklearn.model_selection import train_test_split


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
        nargs = '?',
        default= None,
        help="Specify model filename (without extension) or 'new' for timestamped new model."
    )
    args = parser.parse_args()

    if args.model_name is None:
        # 인수가 없으면 기본 모델 파일 사용
        model_file = "models/catboost_no_one_hot.cbm"
        submisson_name = "baseline"
    elif args.model_name.lower() == "new":
        # 'new'라는 인수를 받으면 현재 시간으로 파일명 생성
        today = datetime.now().strftime('%m%d_%H%M%S')
        model_file = f"models/catboost_{today}.cbm"
        submisson_name = "{today}"
    else:
        # 사용자가 입력한 모델명으로 파일명 결정 (확장자 .cbm 추가)
        model_file = f"models/{args.model_name}.cbm"
        submisson_name = f"{args.model_name}"

    print(f"[INFO] Model file will be: {model_file}")

    return model_file, submisson_name

def main():
    # 0. 터미널에서 모델명 가져오기 (중요 X, 그냥 기능 용도)
    model_file, submisson_name = parsing()
    
    # 1. 데이터 불러오기
    data = pd.read_csv(train_path)
    
    # 2. 데이터 분리 (Stratified 해서 시술 유형 불균형 해결 )
    # X = data.drop(columns = ['임신 성공 여부'])
    # y = data['임신 성공 여부']
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=73, shuffle= True, stratify=X['임신 성공 여부'])
    # X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = 0.5, random_state = 73, shuffle= True, stratify=X_temp['임신 성공 여부'])
    
    X_train, X_valid,  y_train, y_valid, = split_dataset(data)
    
    # 3. 파이프라인 로드 or 학습
    if os.path.exists(model_file) :
        pipeline = CatBoostPipeline.load_pipeline(model_file)
    else : 
        print("새 파이프라인 학습 중")
        pipeline = CatBoostPipeline()
        print(X_train.shape, y_train.shape)
        pipeline.fit(X_train, y_train, X_valid, y_valid)
        pipeline.save_pipeline(model_file)

    
    # 4. 모델 평가
    evaluator = Evaluator()
    print("[Validation Set]")
    X_val_pred_proba = pipeline.predict_proba(X_valid)[:, 1]
    evaluator.evaluate_by_prob(y_valid, X_val_pred_proba)

    pipeline_final = CatBoostPipeline()
    pipeline_final.fit(data.drop(columns=['임신 성공 여부']), data['임신 성공 여부'])
    pipeline_final.save_pipeline(model_file)

    # 제출 파일 새성
    submission = Submisson(pipeline_final.preprocessor)
    today = datetime.now().strftime('%m%d_%H%M%S')
    submission_file = f'submission/submission_{submisson_name}_{today}.csv'
    submission.gen_submit(pipeline_final.model, test_file_path= test_path, output_file= submission_file)

    print(f"[INFO] Model file : {model_file}")
    print(f"[INFO] Submission file : {submission_file}")

    

if __name__ == '__main__':
    main()
