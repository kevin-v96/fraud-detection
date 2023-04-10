# Detecting fraudulent transactions in financial data

In this project, I clean some bank transaction data, explore it to engineer features, and use it to train one of the unsupervised anomaly detection models available in sklearn. Models tried - Isolation Forest, Local Outlier Factor, Autoencoder.

Data cleaning code is in the data folder, preprocessing code (for the preprocessing that sagemaker does as the first step of the pipeline) 
is in the preprocessing folder.

model_selection.ipynb has the code where I tried out different models, and sagemaker_deployment_code.ipynb is the main code that I'm running on SageMaker
after uploading train.py, preprocess.py and flagright-test-transactions.json to an S3 bucket.

I used sagemaker and S3 for deployment because deploying a model on Sagemaker is as simple as training it using an image for your particular framework already availabe on SageMaker, uploading the model artifacts to S3, and deploying an endpoint using this model. The Sagemaker endpoint you get thus can be used to get inference on samples send with curl or Postman, but it needs access keys for an AWS Signature. 

To make a public-facing REST API, I used AWS API Gateway to make a REST Endpoint. API Gateway gets the request from the client, transforms the request data according to mapping templates we define, sends it to the assigned backend (in this case Sagemaker), gets back the response, transforms the response according to another mapping template, and then sends the response back to the client. This REST endpoint can be defined to not require authorization and thus be opened to the public.
