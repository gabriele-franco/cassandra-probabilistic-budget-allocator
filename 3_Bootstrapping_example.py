from utils import *
from cassandra.cas_functions import import_model
from cassandra.cas_functions import *
from cassandra.cas_allocator import *

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
paid_media = json_model['InputCollect']['paid_media_vars']
date_var = json_model['InputCollect']['date_var'][0]
window_start = json_model['InputCollect']['window_start'][0]
window_end = json_model['InputCollect']['window_end'][0]
date_max = ridge_result['date_week'].max()

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


exp_spend_unit_total = 18124
init_spend_unit = [0.0, 496.955, 275.7, 0.0, 0.0, 540.9200000000001, 32.294999999999995, 947.98, 2014.315, 223.0, 0.0, 0.0]
period_to_estimate = 4
channel_constr_low_sorted = [0, 367.3598589405508, 214.83203998480366, 0, 0, 387.9317747252748, 23.420789163628577,
                             801.7245691391942, 1547.317561793812, 168.19980933924222, 0, 0]
channel_constr_up_sorted = [175.12832298136647, 626.5501410594492, 336.5679600151963, 0, 197.54807453416151,
                            693.9082252747254, 41.16921083637141, 1094.235430860806, 2481.3124382061883,
                            277.8001906607578, 377.5833333333333, 40.46204968944099]
channelConstrMeanSorted = [(low + up) / 2 for low, up in zip(channel_constr_low_sorted, channel_constr_up_sorted)]

paid_media_spends = [channelConstrMeanSorted[i] if spend < channel_constr_low_sorted[i] or spend > channel_constr_up_sorted[i] else spend for i, spend in enumerate(init_spend_unit)]


eval_list = {
        'coefsFiltered': [summary_dict[col]['coef'] for col in json_model['InputCollect']['paid_media_vars']],
        'alphas': [summary_dict[col]['alphas'] for col in json_model['InputCollect']['paid_media_vars']],
        'gammas': [summary_dict[col]['gammas'] for col in json_model['InputCollect']['paid_media_vars']],
        'shapes': [summary_dict[col]['shapes'] for col in json_model['InputCollect']['paid_media_vars']],
        'scales': [summary_dict[col]['scales'] for col in json_model['InputCollect']['paid_media_vars']],
        'expSpendUnitTotal': exp_spend_unit_total
    }

paid_media_spend_traspose = df[json_model['InputCollect']['paid_media_vars']].transpose().values.tolist()
target_value = None


def generate_parameter_variation(eval_list, noise_level=0.01):
    noisy_eval_list = {}

    # Iterate over each key in the eval_list except 'expSpendUnitTotal'
    for key, values in eval_list.items():
        if key != 'expSpendUnitTotal':
            # Apply noise to each value in the list
            noisy_values = [
                value + np.random.normal(0, noise_level * value) for value in values
            ]
            noisy_eval_list[key] = noisy_values
        else:
            # Copy expSpendUnitTotal without noise
            noisy_eval_list[key] = eval_list[key]

    return noisy_eval_list



def generate_n_eval_list(eval_list, n):
    evel_list_list = []

    for i in range(0,n):
        evel_list_list.append(generate_parameter_variation(eval_list))

    return evel_list_list


def generate_distribution_of_optimal_budget():
    eval_list = {
        'coefsFiltered': [summary_dict[col]['coef'] for col in json_model['InputCollect']['paid_media_vars']],
        'alphas': [summary_dict[col]['alphas'] for col in json_model['InputCollect']['paid_media_vars']],
        'gammas': [summary_dict[col]['gammas'] for col in json_model['InputCollect']['paid_media_vars']],
        'shapes': [summary_dict[col]['shapes'] for col in json_model['InputCollect']['paid_media_vars']],
        'scales': [summary_dict[col]['scales'] for col in json_model['InputCollect']['paid_media_vars']],
        'expSpendUnitTotal': exp_spend_unit_total
    }

    evel_list_list = generate_n_eval_list(eval_list, 100)
    df_optim_list = []
    for eval_list in evel_list_list:
        budget_allocator_spend = budget_allocation(paid_media_spends, channel_constr_low_sorted, channel_constr_up_sorted,
                                                   period_to_estimate, 700, 1e-10, eval_list, paid_media_spend_traspose,
                                                   period_to_estimate, target_value, 'R')

        df_optim = append_optimal_spend({'df': df, 'paid_media': paid_media, 'date_var': date_var},
                             budget_allocator_spend, period_to_estimate, date_max)

        df_optim['date_week'] = df['date_week']
        extended_window_end = df_optim['date_week'].max()
        df_optim_adstock = apply_adstock_transformation(df_optim, paid_media, summary_dict, date_var)
        df_optim_saturation, df_optim_adstock_filtered = apply_saturation_transformation(df_optim_adstock,
                                                                                         paid_media,
                                                                                         summary_dict,
                                                                                         date_var,
                                                                                         window_start, extended_window_end)
        df_optim_decomp = apply_decomp_transformation(df_optim_saturation, paid_media, summary_dict, date_var)

        df_optim_list.append(df_optim_decomp)

    # can now run statistical analysis on the output predicted revenue
    return df_optim_list






if __name__=='__main__':
    generate_distribution_of_optimal_budget()
