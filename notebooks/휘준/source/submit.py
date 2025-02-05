import pandas as pd

class Submisson :
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


    def gen_submit(self, model, test_file_path, output_file = 'submission/test_submisson.csv'):
        test_data = pd.read_csv(test_file_path)
        X_test = self.preprocessor.transform(test_data) # ID를 뺼까 

        predictions = model.predict_proba(X_test)[:, 1]

        submission_df = pd.read_csv('/Users/hj/projects/Aimers/data/raw/sample_submission.csv')
        submission_df['probability'] = predictions 
        submission_df.to_csv(output_file, index=False)

        print(f"Submission file saved as: {output_file}")

