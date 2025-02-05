from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

class Evaluator:
    def __init__(self):

        pass

    def evaluate(self, model, X_test, y_test):

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, preds)

        # 결과 출력
        print("Accuracy: {:.4f}".format(acc))
        print("ROC-AUC: {:.4f}".format(auc))
        print("\nClassification Report:\n", classification_report(y_test, preds))
        
        return acc, auc