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

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)


def discretize_date(current_date, t):
    current_date = str(current_date)[:-7]
    cdate = datetime.strptime(current_date, '%Y-%m-%d %H:%M:%S')
    if t == 'hour_sin':
        return np.sin(2 * np.pi * cdate.hour/24.0)
    if t == 'hour_cos':
        return np.cos(2 * np.pi * cdate.hour/24.0)
    if t == 'day_sin':
        return np.sin(2 * np.pi * cdate.timetuple().tm_yday/365.0)
    if t == 'day_cos':
        return np.cos(2 * np.pi * cdate.timetuple().tm_yday/365.0)
    if t == 'week_day_sin':
        return np.sin(2 * np.pi * cdate.timetuple().tm_yday/7.0)
    if t == 'week_day_cos':
        return np.cos(2 * np.pi * cdate.timetuple().tm_yday/7.0)


def preprocess_data(data):
    if type(data) == pd.DataFrame:
        features = pd.DataFrame()
        features['state'] = data.get('transactionState')
        features['transactionId'] = data.get('transactionId')
        features['originUserId'] = data.get('originUserId')
        features['destinationUserId'] = data.get('destinationUserId')
        features['destinationCountry'] = data['destinationAmountDetails'].map(lambda x: x['country'])
        features['destinationCurrency'] = data['destinationAmountDetails'].map(lambda x: x['transactionCurrency'])
        features['destinationAmount'] = data['destinationAmountDetails'].map(lambda x: x['transactionAmount'])
        features['originCountry'] = data['originAmountDetails'].map(lambda x: x['country'])
        features['originCurrency'] = data['originAmountDetails'].map(lambda x: x['transactionCurrency'])
        features['originAmount'] = data['originAmountDetails'].map(lambda x: x['transactionAmount'])
        features['destinationMethod'] = data['destinationPaymentDetails'].map(lambda x: x['method'])
        features['originMethod'] = data['originPaymentDetails'].map(lambda x: x['method'])
        features.fillna('N/A', inplace = True)

        features['datetime'] = data['timestamp'].map(lambda x: datetime.fromtimestamp(int(x['$numberLong']) / 1000))
        date_types = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'week_day_sin', 'week_day_cos']
        for dt in date_types:
            features[dt] = features['datetime'].apply(lambda x : discretize_date(x, dt))


        features.drop(columns = ['datetime'], inplace = True)
        
    else:
        print(data)
        print(type(data))
        features = {}
        try:
            features['destinationCountry'] = data['destinationAmountDetails']['country']
        except KeyError:
            features['destinationCountry'] = None
        try:
            features['destinationCurrency'] = data['destinationAmountDetails']['transactionCurrency']
        except KeyError:
            features['destinationCurrency'] = None
        try:
            features['destinationAmount'] = data['destinationAmountDetails']['transactionAmount']
        except KeyError:
            features['destinationAmount'] = None
        try:
            features['originCountry'] = data['originAmountDetails']['country']
        except KeyError:
            features['originCountry'] = None
        try:
            features['originCurrency'] = data['originAmountDetails']['transactionCurrency']
        except KeyError:
            features['originCurrency'] = None
        try:
            features['originAmount'] = data['originAmountDetails']['transactionAmount']
        except KeyError:
            features['originAmount'] = None
        try:
            features['destinationMethod'] = data['destinationPaymentDetails']['method']
        except KeyError:
            features['destinationMethod'] = None
        try:
            features['originMethod'] = data['originPaymentDetails']['method']
        except KeyError:
            features['originMethod'] = None

        features['state'] = data.get('transactionState')
        features['transactionId'] = data.get('transactionId')
        features['originUserId'] = data.get('originUserId')
        features['destinationUserId'] = data.get('destinationUserId')
        try:
            features['datetime'] = datetime.fromtimestamp(int(data['timestamp']['$numberLong']) / 1000)
            date_types = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'week_day_sin', 'week_day_cos']
            for dt in date_types:
                features[dt] = discretize_date(features['datetime'], dt)
            features.pop('datetime')
        except KeyError:
            warnings.warn("No timestamp provided")
        
    return features

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

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
    concat_data = pd.concat(raw_data)

    features = preprocess_data(concat_data)

    # This section is adapted from the scikit-learn example of using preprocessing pipelines:
    #
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    #
    # We will train our anomaly detection model with the following features:
    # Numeric Features:
    # - timestamp:  The timestamp of the time when the transaction was made
    # - amount: Amount of the transaction
    # Categorical Features:
    # - sex: categories encoded as strings {'M', 'F', 'I'} where 'I' is Infant
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler())

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(handle_unknown = 'ignore'))

    preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_exclude="object")),
            ("cat", categorical_transformer, make_column_selector(dtype_include="object"))], sparse_threshold = 0)

    preprocessor.fit(features)

    joblib.dump(preprocessor, os.path.join(args.model_dir, "model.joblib"))

    print("saved model!")


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
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either label encoded or standardized
    """
    features = preprocess_data(input_data)
    transformed_features = model.transform(features)

    return transformed_features


def model_fn(model_dir):
    """Deserialize fitted model
    """
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor