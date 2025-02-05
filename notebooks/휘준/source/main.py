
from preprocessing import Preprocessor
from model import Modeler
from evaluate import Evaluator
from submit import Submisson
from sklearn.model_selection import train_test_split
import os

# 파일 경로 
train_path = '/Users/hj/projects/Aimers/data/raw/train.csv'
test_path = '/Users/hj/projects/Aimers/data/raw/test.csv'

def main():
    # 1. 전처리
    preproc =  Preprocessor(train_path)
    X, y = preproc.fit_transform()

    # 2. 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=73)

    # 3. 모델 학습 (기존 모델 사용 / 모델 새로 학습 )
    model_file = "models/catboost_model_2.cbm"
    model = Modeler()

    if os.path.exists(model_file):
        print("저장된 모델 사용 중")
        model.model.load_model(model_file)
    
    else: 
        print("모델 새로 학습 중")
        model.train(X_train, y_train)
        model.model.save_model(model_file)
    
    # 4. 모델 평가
    evaluator = Evaluator()
    evaluator.evaluate(model, X_test, y_test)

    # 5. 제출본 만들기
    from datetime import datetime
    submission = Submisson(preproc)
    today = datetime.now().strftime('%m%d_%H%M%S')

    submission.gen_submit(model, test_file_path= test_path, output_file= f'submisson/submisson_{today}.csv')

if __name__ == '__main__':
    main()