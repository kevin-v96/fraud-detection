from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil
import argparse
import csv
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

from sklearn.ensemble import IsolationForest

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

def preprocess_data(train_data):
    features_list = train_data["instances"]
    features = [f["features"] for f in features_list]
    original_features = pd.DataFrame(np.array(features))

    return original_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this example we are just including one hyperparameter.
    parser.add_argument('--max_samples', type=str, default='auto')

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [ pd.read_json(file) for file in input_files ]
    train_data = pd.concat(raw_data)


    # Now use scikit-learn's IsolationForest to train the model.
    clf = IsolationForest(random_state = 42)
    original_features = preprocess_data(train_data)
    clf = clf.fit(original_features)

    # Save the model
    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def input_fn(input_data, content_type):
    """Parse input data payload

    """
    if content_type == 'application/json':
        df = pd.read_json(input_data)

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """

    if accept == "application/json":
        return worker.Response(json.dumps(prediction), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def model_fn(model_dir):
    """Deserialize and return fitted model

    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def predict_fn(input_data, model):
    """Preprocess input data and run the model on it
    """

    input_data = preprocess_data(input_data)
    predictions = model.predict(input_data)
    scores = model.score_samples(input_data)
    result = {}
    result["predictions"] = predictions.tolist()
    result["scores"] = scores.tolist()

    return result