import pandas as pd
import json

cassandra_folder = 'cassandra/'

def read_holidays():
    df = pd.read_csv(cassandra_folder + 'dataset-holidays.csv')
    return df

def read_dataset_ms():
    df = pd.read_csv(cassandra_folder + 'dataset-ms.csv')
    return df

def read_decomp_ms():
    df = pd.read_csv(cassandra_folder+ 'decomp-ms.csv', index_col=0)
    return df

def read_json_model():
    json_model = json.load(open(cassandra_folder + '/model-3_623_8.json'))
    return json_model