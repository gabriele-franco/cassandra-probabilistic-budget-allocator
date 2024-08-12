from utils import *
from cassandra.cas_functions import import_model

df = read_dataset_ms()
dc = read_decomp_ms()
json_model = read_json_model()
df_holidays = read_holidays()

common_dates = set(df['date_week']).intersection(set(dc['ds']))
common_cols = set(df.columns).intersection(set(dc.columns))
cols_only_dc = set(dc.columns) -  set(df.columns)
cols_only_df = set(df.columns) -  set(dc.columns)

ridge_model, ridge_result, df_alldecomp_matrix, df_adstock, df_saturation, summary_dict, df_prophet = import_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive', ridge_size=0.2, ridge_positive=True, ridge_random_state=42, type_of_use='import')

all_vars = json_model['InputCollect']['all_ind_vars']
all_media =json_model['InputCollect']['all_media']

ridge_coefs = [value['coef'] for key, value in summary_dict.items() if key != 'intercept']
ridge_keys = [key for key, value in summary_dict.items()]

dfc = df_saturation.copy()
dfc.reset_index(inplace=True, drop=True)
for var in all_vars:
    if var not in dfc.columns:
        dfc[var] = list(df_prophet[var])
dfc = dfc[all_vars]
for i in range(0,len(dfc.columns)):
    col = dfc.columns[i]
    key = ridge_keys[i]
    coeff = ridge_coefs[i]
    dfc[col] = dfc[col]*coeff

print('Ridge prediction match')
print((dfc.sum(axis=1)-ridge_result['prediction']).sum())