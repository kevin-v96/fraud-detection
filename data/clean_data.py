import pandas as pd
import numpy as np
from datetime import datetime

#this data cleaning is was used during data exploration and is nearly the same as the one in the preprocessing step of the pipeline

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


df = pd.read_json("flagright-test-transactions.json")

features = pd.DataFrame()
features['destinationCountry'] = df['destinationAmountDetails'].map(lambda x: x['country'])
features['destinationCurrency'] = df['destinationAmountDetails'].map(lambda x: x['transactionCurrency'])
features['destinationAmount'] = df['destinationAmountDetails'].map(lambda x: x['transactionAmount'])
features['originCountry'] = df['originAmountDetails'].map(lambda x: x['country'])
features['originCurrency'] = df['originAmountDetails'].map(lambda x: x['transactionCurrency'])
features['originAmount'] = df['originAmountDetails'].map(lambda x: x['transactionAmount'])
features['state'] = df['transactionState']
features['destinationMethod'] = df['destinationPaymentDetails'].map(lambda x: x['method'])
features['originMethod'] = df['originPaymentDetails'].map(lambda x: x['method'])
features['transactionId'] = df['transactionId']
features['originUserId'] = df['originUserId']
features['destinationUserId'] = df['destinationUserId']
features.fillna('N/A', inplace = True)

features['datetime'] = df['timestamp'].map(lambda x: datetime.fromtimestamp(int(x['$numberLong']) / 1000))
date_types = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'week_day_sin', 'week_day_cos']
for dt in date_types:
    features[dt] = features['datetime'].apply(lambda x : discretize_date(x, dt))


features.drop(columns = ['datetime'], inplace = True)
features.to_csv("flagright.csv")