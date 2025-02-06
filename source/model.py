from catboost import CatBoostClassifier

class Modeler :
    def __init__(self, params = None):
        if params is None:
            params = {
                'iterations': 3000,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'verbose': 100,
                'random_seed': 73,
                'early_stopping_rounds': 50
            }
        self.model = CatBoostClassifier(**params)


    def train(self, X_train, y_train, X_valid = None, y_valid = None) :
        eval_set = None
        if X_valid is not None and y_valid is not None:
            eval_set = (X_valid, y_valid)
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            use_best_model=True,
            verbose=True
        )

    def predict(self, X):

        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        입력 데이터에 대한 예측 확률을 반환합니다.
        
        :param X: 예측할 데이터의 feature 행렬
        :return: 예측 확률 배열
        """
        return self.model.predict_proba(X)