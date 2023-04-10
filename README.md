# Detecting fraudulent transactions in financial data

Data cleaning code is in the data folder, preprocessing code (for the preprocessing that sagemaker does as the first step of the pipeline) 
is in the preprocessing folder.

model_selection has the code where I tried out different models, sagemaker_deployment_code.ipynb is the main code that I'm running on SageMaker
after uploading train.py, preprocess.py and flagright-test-transactions.json to an S3 bucket.
