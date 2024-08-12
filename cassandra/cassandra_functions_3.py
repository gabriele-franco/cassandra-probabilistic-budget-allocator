from prophet import Prophet
from scipy.stats import weibull_min, norm, t
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import pandas as pd
import numpy as np
import nevergrad as ng
import random

def bootci(samp, share_spend, boot_n, seed=1):
    random.seed(seed)
    n = len(samp)
    ci_percentage = 1 - share_spend
    ci_width = np.mean(samp) * ci_percentage
    
    # Se abbiamo un solo campione, non possiamo fare il bootstrap
    if n <= 1:
        return {'boot_means': np.array(samp), 'ci': (np.mean(samp) - ci_width, np.mean(samp) + ci_width), 'se': 0}
    
    samp_n = len(samp)
    samp_mean = np.mean(samp)
    boot_sample = np.random.choice(samp, size=(boot_n, samp_n), replace=True)
    boot_means = np.mean(boot_sample, axis=1)
    se = np.std(boot_means, ddof=1)

    # Calculating the margin of error
    me = t.ppf(0.975, samp_n - 1) * se
    samp_me = me * np.sqrt(samp_n)
    ci_low = samp_mean - samp_me
    ci_up = samp_mean + samp_me
    boot_mean = np.mean(boot_means)

    if boot_mean == ci_low:
        ci_low = samp_mean - ci_width
    if boot_mean == ci_up:
        ci_up = samp_mean + ci_width

    ci = (ci_low, ci_up)

    return {'boot_means': boot_means, 'ci': ci, 'se': se}

def confidence_calcs(xDecompAgg, share_spend, all_paid, dep_var_type, k,
                     boot_n=1000, sim_n=10000):
    # Assuming the required preprocessing steps on xDecompAgg and cls are similar to those in the R code
    df_clusters_outcome = xDecompAgg[xDecompAgg['total_spend'].notna()]
    df_clusters_outcome = df_clusters_outcome[['solID', 'cluster', 'rn', 'roi_total', 'cpa_total']]
    df_clusters_outcome = df_clusters_outcome[df_clusters_outcome['cluster'].notna()]
    df_clusters_outcome = df_clusters_outcome.sort_values(by=['cluster', 'rn'])

    cluster_collect = []
    chn_collect = []
    sim_collect = []

    for j in k:
        df_outcome = df_clusters_outcome[df_clusters_outcome['cluster'] == j]

        if len(df_outcome['solID'].unique()) < 3:
            print(f"Warning: Cluster {j} does not contain enough models to calculate CI")

            for i in all_paid:

                if dep_var_type in ['O', 'C', 'N']:
                    df_chn = df_outcome[(df_outcome['rn'] == i) & np.isfinite(df_outcome['cpa_total'])]
                    v_samp = df_chn['cpa_total'].values 
                else:
                    df_chn = df_outcome[df_outcome['rn'] == i]
                    v_samp = df_chn['roi_total'].values

                if len(v_samp) > 0:
                    boot_mean = np.mean(v_samp)

                    df_chn = df_chn.copy()
                    df_chn['ci_low'] = boot_mean
                    df_chn['ci_up'] = boot_mean
                    df_chn['n'] = len(v_samp)
                    df_chn['boot_se'] = 0
                    df_chn['boot_mean'] = boot_mean
                    df_chn['cluster'] = j
                    chn_collect.append(df_chn)

                    x_sim = np.random.normal(boot_mean, 0, sim_n)
                    y_sim = norm.pdf(x_sim, boot_mean, 0)
                
                else:
                    boot_mean = 0
                    x_sim = np.array([])
                    y_sim = np.array([])

                sim_collect.append(pd.DataFrame({'cluster': j, 'rn': i, 'n': len(v_samp),
                                                    'boot_mean': boot_mean, 'x_sim': x_sim, 'y_sim': y_sim}))
        else:
            for i in all_paid:
                if dep_var_type in ['O', 'C', 'N']:
                    df_chn = df_outcome[(df_outcome['rn'] == i) & np.isfinite(df_outcome['cpa_total'])]
                    v_samp = df_chn['cpa_total'].to_numpy()
                else:
                    df_chn = df_outcome[df_outcome['rn'] == i]
                    v_samp = df_chn['roi_total'].to_numpy()  
                
                if len(v_samp) > 0:
                    boot_res = bootci(v_samp, share_spend[i], boot_n=boot_n)
                    boot_mean = np.mean(boot_res['boot_means'])
                    boot_se = boot_res['se']
                    ci_low = max(0, boot_res['ci'][0])
                    ci_up = boot_res['ci'][1]

                    # Collect loop results
                    df_chn = df_chn.copy()
                    df_chn['ci_low'] = ci_low
                    df_chn['ci_up'] = ci_up
                    df_chn['n'] = len(v_samp)
                    df_chn['boot_se'] = boot_se
                    df_chn['boot_mean'] = boot_mean
                    df_chn['cluster'] = j
                    chn_collect.append(df_chn)

                    x_sim = np.random.normal(boot_mean, boot_se, sim_n)
                    y_sim = norm.pdf(x_sim, boot_mean, boot_se)

                else:
                    boot_mean = 0
                    x_sim = np.array([])
                    y_sim = np.array([])
                
                sim_collect.append(pd.DataFrame({'cluster': j, 'rn': i, 'n': len(v_samp),
                                                 'boot_mean': boot_mean, 'x_sim': x_sim, 'y_sim': y_sim}))
                
        #cluster_collect.append({'chn_collect': chn_collect, 'sim_collect': sim_collect})
        cluster_collect.append({'chn_collect': chn_collect})

    # Aggregating simulation data
    '''sim_collect = pd.concat([pd.concat(x['sim_collect']) for x in cluster_collect])
    sim_collect['cluster_title'] = sim_collect.apply(lambda row: f"Cl.{row['cluster']} (n={row['n']})", axis=1)'''

    # Aggregating CI data
    df_ci = pd.concat([pd.concat(x['chn_collect']) for x in cluster_collect])
    df_ci['cluster_title'] = df_ci.apply(lambda row: f"Cl.{row['cluster']} (n={row['n']})", axis=1)
    df_ci = df_ci.groupby(['rn', 'cluster_title', 'cluster']).agg({
        'n': 'first', 'boot_mean': 'first', 'boot_se': 'first',
        'ci_low': 'first', 'ci_up': 'first'
    }).reset_index()

    return {'df_ci': df_ci, 'sim_collect': sim_collect, 'boot_n': boot_n, 'sim_n': sim_n}


def refresh_model(df, df_holidays, json_model, num_trials, iterations):
    init_result, init_model, init_summary_dict, init_df_saturation_ridge, init_lambda_value, init_ridge_intercept, init_rsq, init_nrmse, init_rssd, init_mape = import_model(json_model, df, df_holidays, type_of_use='refresh')

    def robyn_model_obj(**parameters):
        new_summary_dict = {}
        
        for key, value in init_summary_dict.items():
            if key + '_alphas' in parameters.keys():
                new_summary_dict[key] = {'coef': parameters[key], 'alphas': parameters[key + '_alphas'], 'gammas': parameters[key + '_gammas'], 'shapes': parameters[key + '_shapes'], 'scales': parameters[key + '_scales']}
            else:
                new_summary_dict[key] = {'coef': parameters[key]}
            
        new_json_model = create_summary_and_hyper_values(json_model, new_summary_dict, init_lambda_value)

        new_result, new_model, new_summary_dict_obj, new_df_saturation_ridge, new_lambda_value, new_ridge_intercept, new_rsq, new_nrmse, new_rssd, new_mape = import_model(new_json_model, df, df_holidays, type_of_use='refresh')
        
        if init_mape:
            if new_mape == 0:
                new_mape = 0.01
            return [new_nrmse, new_rssd, new_mape]
        else:
            return [new_nrmse, new_rssd]
        
    date_var = json_model['InputCollect']['date_var'][0]
    window_end = json_model['InputCollect']['window_end'][0]
    window_start = json_model['InputCollect']['refreshAddedStart'][0]
    
    df_window = df.loc[(df[date_var] >= window_start) & (df[date_var] <= window_end)]
    
    paid_media_vars_and_signs = dict(zip(json_model['InputCollect']['paid_media_spends'], json_model['InputCollect']['paid_media_signs']))
    if 'organic_vars' in json_model['InputCollect']:
        organic_vars_and_signs = dict(zip(json_model['InputCollect']['organic_vars'], json_model['InputCollect']['organic_signs']))
    else:
        organic_vars_and_signs = {}
    if 'context_vars' in json_model['InputCollect']:
        context_vars_and_signs = dict(zip(json_model['InputCollect']['context_vars'], json_model['InputCollect']['context_signs']))
    else:
        context_vars_and_signs = {}
    if 'prophet_vars' in json_model['InputCollect']:
        prophet_vars_and_signs = dict(zip(json_model['InputCollect']['prophet_vars'], json_model['InputCollect']['prophet_signs']))
    else:
        prophet_vars_and_signs = {}
    
    vars_and_signs = paid_media_vars_and_signs | organic_vars_and_signs | context_vars_and_signs | prophet_vars_and_signs | {'intercept': 'positive'}

    # Define a list to store the best models from each trial
    models = []

    dict_pareto_aggregated_media = {
        'cluster': [],
        'rn':[],
        'solID':[],
        'total_spend':[],
        'total_response':[],
        'cpa_total':[],
        'roi_total':[]
    }
    count_trials = 1

    # Run the optimization process for each trial
    for trial in range(num_trials):
        # Set a random seed for reproducibility
        np.random.seed(trial)

        parameters=create_parametrization(init_summary_dict, vars_and_signs, type_of_use='instrumentation')       
        hyper_updated=create_parametrization(init_summary_dict, vars_and_signs, type_of_use='hyper_updated')

        instrum = ng.p.Instrumentation(**parameters)
        optimizer= ng.optimizers.TwoPointsDE(instrum, budget=iterations)

        optimizer.minimize(robyn_model_obj)

        # Store all models from this trial
        if optimizer.pareto_front():
            trial_models = [col.value[1] for col in optimizer.pareto_front()]
        else:
            trial_models = [optimizer.recommend().value[1]]
        
        count_models = 1

        for trial_model in trial_models:
            new_summary_dict = {}

            for key, value in init_summary_dict.items():
                if key + '_alphas' in parameters.keys():
                    new_summary_dict[key] = {'coef': trial_model[key], 'alphas': trial_model[key + '_alphas'], 'gammas': trial_model[key + '_gammas'], 'shapes': trial_model[key + '_shapes'], 'scales': trial_model[key + '_scales']}
                else:
                    new_summary_dict[key] = {'coef': trial_model[key]}
                
            json_model = create_summary_and_hyper_values(json_model, new_summary_dict, init_lambda_value)
            
            result, model, summary_dict_obj, df_saturation_ridge, lambda_value, ridge_intercept, rsq, nrmse, rssd, mape = import_model(json_model, df, df_holidays, type_of_use='refresh')
            
            summary_dict_obj["rsq"] = rsq
            summary_dict_obj["nrmse"] = nrmse
            summary_dict_obj["decomp_rssd"] = rssd
            summary_dict_obj["mape"] = mape

            df_saturation_ridge_model_window = df_saturation_ridge.loc[(df_saturation_ridge[date_var] >= window_start) & (df_saturation_ridge[date_var] <= window_end)]
            
            # Add the models from this trial to the list
            models.append(summary_dict_obj)

            for media in json_model['InputCollect']['paid_media_spends']:
                total_spend_model = df_window[media].sum()
                total_response_model = sum([value * summary_dict_obj[media]['coef'] for value in df_saturation_ridge_model_window[media]])
                dict_pareto_aggregated_media['cluster'].append(1)
                dict_pareto_aggregated_media['rn'].append(media)
                dict_pareto_aggregated_media['solID'].append(str(count_trials) + '_' + str(count_models))
                dict_pareto_aggregated_media['total_spend'].append(total_spend_model)
                dict_pareto_aggregated_media['total_response'].append(sum([value * summary_dict_obj[media]['coef'] for value in df_saturation_ridge[media]]))
                dict_pareto_aggregated_media['cpa_total'].append(total_spend_model / total_response_model if int(total_response_model) != 0 else 0)
                dict_pareto_aggregated_media['roi_total'].append(total_response_model / total_spend_model if int(total_spend_model) != 0 else 0)

            count_models += 1
        count_trials += 1

    if json_model['InputCollect']['dep_var_type'][0] == 'conversion':
        dep_var_type = 'C'
    else:
        dep_var_type = 'R'
    
    df_pareto_aggregated = pd.DataFrame(dict_pareto_aggregated_media)
    share_spend = {}
    total_spend = df_window[json_model['InputCollect']['paid_media_spends']].sum().sum()

    for media in json_model['InputCollect']['paid_media_spends']:
        share_spend[media] = df_window[media].sum() / total_spend
        
    ci_list = confidence_calcs(df_pareto_aggregated, share_spend, json_model['InputCollect']['paid_media_spends'], dep_var_type, 1)

    bootstrap_df = ci_list['df_ci'].reset_index(drop=True)

    # Calculate the minimum and maximum values for each error metric
    nrmse_values = [model['nrmse'] for model in models]
    decomp_rssd_values = [model['decomp_rssd'] for model in models]
    mape_values = [model['mape'] for model in models]

    nrmse_min, nrmse_max = min(nrmse_values), max(nrmse_values)
    decomp_rssd_min, decomp_rssd_max = min(decomp_rssd_values), max(decomp_rssd_values)
    
    # Check if there's MAPE - Which means calibration used
    if mape_values[0] is not None:
        mape_min, mape_max = min(mape_values), max(mape_values)
    else:
        mape_min, mape_max = None, None

    # Calculate the error score for each model
    for model in models:
        model['error_score'] = compute_error_score(model['nrmse'], model['decomp_rssd'], model['mape'], [1, 1, 1], 
                                                nrmse_min, nrmse_max, decomp_rssd_min, decomp_rssd_max, mape_min, mape_max)

    # Select the model with the smallest error score
    best_model = min(models, key=lambda model: model['error_score'])

    json_model = create_summary_and_hyper_values(json_model, best_model, init_lambda_value)
    result, model, summary_dict_obj, df_saturation_ridge, lambda_value, ridge_intercept, rsq, nrmse, rssd, mape = import_model(json_model, df, df_holidays, type_of_use='refresh')
    
    summary_list = json_model['ExportedModel']["summary"]
    
    df_saturation_ridge_window = df_saturation_ridge.loc[(df_saturation_ridge[date_var] >= window_start) & (df_saturation_ridge[date_var] <= window_end)]
    
    for model_summary in summary_list:
        if model_summary["variable"] in json_model['InputCollect']['paid_media_spends']:
            total_spend_top = df_window[model_summary["variable"]].sum()
            total_response_top = sum([value * summary_dict_obj[model_summary["variable"]]['coef'] for value in df_saturation_ridge_window[model_summary["variable"]]])
    
            if json_model['InputCollect']['dep_var_type'][0] == 'conversion':
                cpa_total_top = total_spend_top / total_response_top if int(total_response_top) != 0 else 0
            else:
                cpa_total_top = total_response_top / total_spend_top if int(total_spend_top) != 0 else 0
        
            model_summary["boot_mean_cassandra"] = bootstrap_df[bootstrap_df['rn'] == model_summary["variable"]]['boot_mean'].iloc[0]
            model_summary["ci_low_cassandra"] = bootstrap_df[bootstrap_df['rn'] == model_summary["variable"]]['ci_low'].iloc[0] if bootstrap_df[bootstrap_df['rn'] == model_summary["variable"]]['ci_low'].iloc[0] < cpa_total_top else cpa_total_top 
            model_summary["ci_up_cassandra"] = bootstrap_df[bootstrap_df['rn'] == model_summary["variable"]]['ci_up'].iloc[0] if bootstrap_df[bootstrap_df['rn'] == model_summary["variable"]]['ci_up'].iloc[0] > cpa_total_top else cpa_total_top
    
    return result, model, summary_dict_obj, summary_list, df_saturation_ridge, best_model, hyper_updated, lambda_value, ridge_intercept, rsq, nrmse, rssd, mape

# Remember to adjust the balance according to your needs.
# For example, if you want to give equal importance to all metrics, you can set balance = [1, 1, 1].
# If you want to give more importance to NRMSE and less to the others, you can set balance = [2, 1, 1], and so on.
# These weights should be normalized to sum to 1, but the compute_error_score function takes care of that for you.
def compute_error_score(nrmse, decomp_rssd, mape=None, balance=None, 
                    nrmse_min=None, nrmse_max=None, decomp_rssd_min=None, decomp_rssd_max=None, mape_min=None, mape_max=None):
    # Normalize the error metrics
    nrmse_n = (nrmse - nrmse_min) / (nrmse_max - nrmse_min)
    decomp_rssd_n = (decomp_rssd - decomp_rssd_min) / (decomp_rssd_max - decomp_rssd_min)
    
    # Calculate the weighted sum of the squared error metrics
    if mape is not None and mape_min is not None and mape_max is not None:
        mape_n = (mape - mape_min) / (mape_max - mape_min)
        error_score = math.sqrt(balance[0]*nrmse_n**2 + balance[1]*decomp_rssd_n**2 + balance[2]*mape_n**2)
    else:
        error_score = math.sqrt(balance[0]*nrmse_n**2 + balance[1]*decomp_rssd_n**2)

    return error_score

def create_summary_and_hyper_values(json_model, summary_dict, lambda_value):
    hyper_values = {}
    summary = []

    for key, value in summary_dict.items():
        if isinstance(value, dict) and 'alphas' in list(value.keys()):
            summary.append({'variable': key, 'coef': value['coef']})
            hyper_values[key + '_alphas'] = [value['alphas']]
            hyper_values[key + '_gammas'] = [value['gammas']]
            hyper_values[key + '_scales'] = [value['scales']]
            hyper_values[key + '_shapes'] = [value['shapes']]
        else:
            if key == 'intercept':
                summary.append({'variable': '(Intercept)', 'coef': value['coef']})
            elif key not in ("rsq", "nrmse", "decomp_rssd", "mape", "error_score"):
                summary.append({'variable': key, 'coef': value['coef']})
    
    hyper_values['lambda'] = [lambda_value]

    json_model['ExportedModel']['summary']=summary
    json_model['ExportedModel']['hyper_values']=hyper_values

    return json_model

def calculate_adjustment_factor(boot_mean, ci_low, ci_up):
    factor_low = abs(boot_mean - ci_low) / boot_mean if boot_mean != 0 else 0.3
    factor_up = abs(boot_mean - ci_up) / boot_mean if boot_mean != 0 else 0.3

    return factor_low, factor_up

def create_summary_dictionary(all_vars, hyper_dict, coef_dict):
    """
    Constructs a summary dictionary containing coefficients and hyperparameters for each variable.

    Args:
        all_vars (list): A list of all variable names.
        hyper_dict (dict): A dictionary containing hyperparameters for variables.
        coef_dict (dict): A dictionary containing coefficients for variables.

    Returns:
        dict: A summary dictionary containing structured information about each variable,
              including coefficients and hyperparameters.
    """
    summary_dict = {}

    for variable in all_vars:
        variable_details = {'coef': coef_dict.get(f'{variable}_coef', 0)}  # Default coef to 0 if not found

        # Check and include hyperparameters if they exist for the variable
        if f'{variable}_alphas' in hyper_dict:
            variable_details.update({
                'alphas': hyper_dict[f'{variable}_alphas'],
                'gammas': hyper_dict[f'{variable}_gammas'],
                'shapes': hyper_dict[f'{variable}_shapes'],
                'scales': hyper_dict[f'{variable}_scales']
            })

        # Check and include bootstrapping metrics if they exist for the variable
        boot_mean_key = f'{variable}_boot_mean_cassandra'
        if boot_mean_key in coef_dict:
            variable_details.update({
                'boot_mean_cassandra': coef_dict[boot_mean_key],
                'ci_up_cassandra': coef_dict[f'{variable}_ci_up_cassandra'],
                'ci_low_cassandra': coef_dict[f'{variable}_ci_low_cassandra']
            })

        summary_dict[variable] = variable_details

    # Handle intercept separately
    intercept_key = '(Intercept)'
    if intercept_key in coef_dict:
        summary_dict['intercept'] = {'coef': coef_dict[intercept_key]}
    else:
        summary_dict['intercept'] = {'coef': 0}

    return summary_dict

#Pass a dict {channel: {coef:value, alphas:value, gammas:value, ecc...}} and use this for create a Scalar value to pass a Instrumentation
def create_parametrization(summary_dict, vars_and_signs, ci_dict={}, type_of_use='instrumentation'):
    parameters = {}

    for key, value in summary_dict.items():
        if 'boot_mean_cassandra' in value: 
            hyper_variance_low, hyper_variance_up = calculate_adjustment_factor(summary_dict[key]['boot_mean_cassandra'], summary_dict[key]['ci_low_cassandra'], summary_dict[key]['ci_up_cassandra'])
            coef_variance_low, coef_variance_up = hyper_variance_low, hyper_variance_up
        else:
            hyper_variance_low, hyper_variance_up = 0.3, 0.3
            coef_variance_low, coef_variance_up = 0.4, 0.4
                
        if 'alphas' in value:
            alpha_low = summary_dict[key]['alphas'] - (summary_dict[key]['alphas'] * hyper_variance_low)
            if alpha_low <= 0.3:
                alpha_low = 0.3
            
            alpha_up = summary_dict[key]['alphas'] + (summary_dict[key]['alphas'] * hyper_variance_up)

            if alpha_up > 2.5 and alpha_low < 2.5:
                alpha_up = 2.5
            elif alpha_up <= alpha_low:
                alpha_up = alpha_low + 0.1

            if type_of_use == 'hyper_updated':
                parameters[f'{key}_alphas']=[alpha_low, alpha_up]

            elif type_of_use == 'instrumentation':
                parameters[f'{key}_alphas']=ng.p.Scalar(lower=alpha_low, upper=alpha_up)
                #print("TODO: Nevergrad")
        
        if 'gammas' in value:
            gamma_low = summary_dict[key]['gammas'] - (summary_dict[key]['gammas'] * hyper_variance_low)
            if gamma_low <= 0.1:
                gamma_low = 0.1
            
            gamma_up = summary_dict[key]['gammas'] + (summary_dict[key]['gammas'] * hyper_variance_up)
            if gamma_up > 0.9 and gamma_low < 0.9:
                gamma_up = 0.9
            elif gamma_up <= gamma_low:
                gamma_up = gamma_low + 0.05

            if gamma_up > 1:
                gamma_up = 1
                gamma_low = gamma_up - 0.05

            if type_of_use == 'hyper_updated':
                parameters[f'{key}_gammas']=[gamma_low, gamma_up]

            elif type_of_use == 'instrumentation':
                parameters[f'{key}_gammas']=ng.p.Scalar(lower=gamma_low, upper=gamma_up)
                #print("TODO: Nevergrad")
        
        if 'shapes' in value:
            #shape_low = summary_dict[key]['shapes']*0.8
            shape_low = summary_dict[key]['shapes'] - (summary_dict[key]['shapes'] * hyper_variance_low)
            if shape_low <= 0:
                shape_low = 0.1
            
            #shape_up = summary_dict[key]['shapes']*1.2
            shape_up = summary_dict[key]['shapes'] + (summary_dict[key]['shapes'] * hyper_variance_up)
            if shape_up > 7:
                shape_up = 7

            if shape_up <= shape_low:
                shape_up = shape_low + 0.5
                
            if type_of_use == 'hyper_updated':
                parameters[f'{key}_shapes']=[shape_low, shape_up]

            elif type_of_use == 'instrumentation':
                parameters[f'{key}_shapes']=ng.p.Scalar(lower=shape_low, upper=shape_up)
                #print("TODO: Nevergrad")
        
        if 'scales' in value:
            #scale_low = summary_dict[key]['scales']*0.8
            scale_low = summary_dict[key]['scales'] - (summary_dict[key]['scales'] * hyper_variance_low)
            if scale_low <= 0:
                scale_low = 0.001
            
            #scale_up = summary_dict[key]['scales']*1.2
            scale_up = summary_dict[key]['scales'] + (summary_dict[key]['scales'] * hyper_variance_up)
            if scale_up > 0.25:
                scale_up = 0.25

            if scale_up <= scale_low:
                scale_up = scale_low + 0.005
                
            if type_of_use == 'hyper_updated':
                parameters[f'{key}_scales']=[scale_low, scale_up]

            elif type_of_use == 'instrumentation':
                parameters[f'{key}_scales']=ng.p.Scalar(lower=scale_low, upper=scale_up)
                #print("TODO: Nevergrad")

        if type_of_use == 'instrumentation':
            if 'coef' in value:
                if summary_dict[key]['coef'] != 0:
                    coef_low = summary_dict[key]['coef'] - (summary_dict[key]['coef'] * coef_variance_low)

                    if coef_low < 0 and vars_and_signs[key] == 'positive':
                        coef_low = 0

                    coef_up = summary_dict[key]['coef'] + (summary_dict[key]['coef'] * coef_variance_up)

                    if coef_up > 0 and vars_and_signs[key] == 'negative':
                        coef_up = 0

                    if coef_up == 0:
                        coef_up = 0.000001
                    
                    if coef_up == coef_low:
                        coef_up = coef_low + 0.001

                    if coef_low < coef_up:
                        parameters[f'{key}']=ng.p.Scalar(lower=coef_low, upper=coef_up)
                    else:
                        parameters[f'{key}']=ng.p.Scalar(lower=coef_up, upper=coef_low)
                else:
                    parameters[f'{key}']=ng.p.Scalar(lower=0, upper=0.000001)
    
    return parameters

def apply_adstock_transformation(df, all_media, summary_dict, date_var):
    """
    Applies adstock transformation to media columns in the dataframe based on provided shapes and scales.

    Args:
        df (pd.DataFrame): Original dataframe containing media columns and a date column.
        all_media (list): List of column names in `df` representing media variables to be adstock transformed.
        summary_dict (dict): Dictionary containing 'shapes' and 'scales' for adstock transformation of each media variable.
        date_var (str): The name of the date column in `df`.

    Returns:
        pd.DataFrame: A new dataframe with adstock-transformed media variables and the original date column.
    """
    df_adstock = df[all_media].copy()
    for col in df_adstock.columns:
        shape, scale = summary_dict[col]['shapes'], summary_dict[col]['scales']
        df_adstock[col] = adstock(df_adstock[col], shape, scale)['x_decayed']
    df_adstock.reset_index(inplace=True, drop=True)
    df_adstock[date_var] = list(df[date_var])
    
    return df_adstock

def apply_saturation_transformation(df_adstock, all_media, summary_dict, date_var, window_start, window_end):
    """
    Filters the adstock-transformed DataFrame for a specified date window and applies saturation transformations
    to media variables using the Hill function based on alphas and gammas from the summary dictionary. Appends
    the original date variable to the resulting DataFrame.

    Args:
        df_adstock (pd.DataFrame): DataFrame containing adstock-transformed media variables and a date column.
        all_media (list): List of column names representing media variables to be saturation transformed.
        summary_dict (dict): Dictionary containing 'alphas' and 'gammas' for saturation transformation of each media variable.
        date_var (str): The name of the date column in `df_adstock`.
        window_start (str): The start of the date window for filtering `df_adstock`.
        window_end (str): The end of the date window for filtering `df_adstock`.

    Returns:
        pd.DataFrame: A new DataFrame with saturation-transformed media variables and the original date column, filtered
                      by the specified date window.
    """
    # Filter adstock DataFrame for the specified date window
    df_adstock_filtered = df_adstock.loc[(df_adstock[date_var] >= window_start) & (df_adstock[date_var] <= window_end)]
    
    # Copy the filtered media columns to a new DataFrame for saturation transformation
    df_saturation = df_adstock_filtered[all_media].copy()
    
    # Apply saturation transformation to each media column
    for col in df_saturation.columns:
        alphas, gammas = summary_dict[col]['alphas'], summary_dict[col]['gammas']
        df_saturation[col] = saturation_hill(df_saturation[col], alphas, gammas)
    
    # Reset index and append the date variable from the filtered DataFrame
    df_saturation.reset_index(inplace=True, drop=True)
    df_saturation[date_var] = list(df_adstock_filtered[date_var])
    
    return df_saturation, df_adstock_filtered

def apply_decomp_transformation(df_saturation, all_media, summary_dict, date_var):
    """
    Apply decomposition transformation to the given dataframe.

    Args:
        df_saturation (pandas.DataFrame): The input dataframe containing saturation data.
        all_media (list): List of media columns to apply the transformation.
        summary_dict (dict): Dictionary containing coefficient values for each media column.
        date_var (str): Name of the date variable column.

    Returns:
        pandas.DataFrame: The transformed dataframe with decomposition applied.
    """
    df_alldecomp = df_saturation[all_media].copy()

    for col in df_alldecomp.columns:
        df_alldecomp[col] = summary_dict[col]['coef'] * df_saturation[col]

    df_alldecomp[date_var] = list(df_saturation[date_var])

    return df_alldecomp

def extract_hyperparameters(hyper_values):
    """
    Extracts hyperparameters from the given JSON model, excluding 'lambda',
    which is assigned to a separate variable.

    Args:
        json_model (dict): The JSON model object containing 'ExportedModel' and 'hyper_values'.

    Returns:
        tuple: A dictionary of hyperparameters and the lambda value extracted from the model.
    """
    hyper_dict = {}
    lambda_value = 0

    # Iterate through hyper_values, excluding 'lambda'
    for key, value in hyper_values.items():
        if key != 'lambda':
            hyper_dict[key] = value[0]  # Extract the first value of the list for each hyperparameter
        else:
            lambda_value = value[0]  # Assign lambda value separately

    return hyper_dict, lambda_value

def extract_coefficients_and_confidence_intervals(summary, paid_media):
    """
    Extracts coefficients and, for paid media variables, bootstrap means and confidence intervals
    from the model's summary.

    Args:
        json_model (dict): The JSON model object containing 'ExportedModel' and its 'summary'.
        paid_media (list): A list of strings representing the names of paid media variables.

    Returns:
        dict: A dictionary containing coefficients and, for paid media variables,
              bootstrap means and confidence intervals.
    """
    coef_dict = {}

    for elem in summary:
        # Always extract coefficients
        coef_key = f"{elem['variable']}_coef"
        coef_dict[coef_key] = elem['coef']

        # For paid media variables, extract additional statistics if available
        if elem["variable"] in paid_media:
            boot_mean_key = f"{elem['variable']}_boot_mean_cassandra"
            ci_up_key = f"{elem['variable']}_ci_up_cassandra"
            ci_low_key = f"{elem['variable']}_ci_low_cassandra"

            # Check for Cassandra-specific or general bootstrapping metrics
            if 'boot_mean_cassandra' in elem:
                coef_dict[boot_mean_key] = elem['boot_mean_cassandra']
                coef_dict[ci_up_key] = elem['ci_up_cassandra']
                coef_dict[ci_low_key] = elem['ci_low_cassandra']
            elif 'boot_mean' in elem:
                coef_dict[boot_mean_key] = elem['boot_mean']
                coef_dict[ci_up_key] = elem['ci_up']
                coef_dict[ci_low_key] = elem['ci_low']

    return coef_dict

def import_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
    prophet_seasonality_mode='additive', ridge_size=0.2, ridge_positive=True, ridge_random_state=42, type_of_use='import'):
    
    date_var = json_model['InputCollect']['date_var'][0]
    dep_var = json_model['InputCollect']['dep_var'][0]
    dep_var_type = json_model['InputCollect']['dep_var_type'][0]
    if 'prophet_vars' in json_model['InputCollect']:
        prophet_vars = json_model['InputCollect']['prophet_vars']
    else:
        prophet_vars = []
    if 'prophet_country' in json_model['InputCollect']:
        prophet_country = json_model['InputCollect']['prophet_country'][0]
    else:
        prophet_country = '-'
        
    day_interval = json_model['InputCollect']['dayInterval'][0]
    interval_type = json_model['InputCollect']['intervalType'][0]
    window_start = json_model['InputCollect']['window_start'][0]
    window_end = json_model['InputCollect']['window_end'][0]
    paid_media = json_model['InputCollect']['paid_media_spends']
    if 'organic_vars' in json_model['InputCollect']:
        organic = json_model['InputCollect']['organic_vars']
    else:
        organic = []
    if 'context_vars' in json_model['InputCollect']:
        context = json_model['InputCollect']['context_vars']
    else:
        context = []
    all_media = json_model['InputCollect']['all_media']
    all_vars = json_model['InputCollect']['all_ind_vars']
    
    if interval_type == 'day':
        prophet_freq='D'
    elif interval_type == 'week':
        prophet_freq='W'
    elif interval_type == 'month':
        prophet_freq='M'

    df_window = df.loc[(df[date_var] >= window_start) & (df[date_var] <= window_end)]
    df_window.reset_index(inplace=True, drop=True)

    ####################################################SECTION PROPHET####################################################
    
    df_copy = df.copy()

    df_prophet = prophet_cassandra(df_copy, df_holidays, date_var, dep_var, prophet_vars, window_start='', window_end='',
            national_holidays_abbreviation=prophet_country, future_dataframe_periods=prophet_future_dataframe_periods, freq=prophet_freq, seasonality_mode=prophet_seasonality_mode)
    
    df_prophet = df_prophet.loc[(df_prophet[date_var] >= window_start) & (df_prophet[date_var] <= window_end)]
    df_prophet.reset_index(inplace=True, drop=True)
    ####################################################SECTION COEFF AND HYPERPARAMETERS####################################################

    #take a coef
    coef_dict = extract_coefficients_and_confidence_intervals(json_model['ExportedModel']['summary'], paid_media)
                
    #take a hyperparameters
    hyper_dict, lambda_value = extract_hyperparameters(json_model['ExportedModel']['hyper_values'])

    summary_dict = create_summary_dictionary(all_vars, hyper_dict, coef_dict)
    
    use_intercept = True if summary_dict['intercept']['coef'] != 0 else False
        
    ####################################################SECTION CREATE ADSTOCK AND SATURATION DATASET####################################################

    df_adstock = apply_adstock_transformation(df, all_media, summary_dict, date_var)

    df_saturation, df_adstock_filtered = apply_saturation_transformation(df_adstock, all_media, summary_dict, date_var, window_start, window_end)

    ####################################################SECTION RIDGE REGRESSION####################################################
    
    df_saturation_ridge = df_saturation.copy()
    df_saturation_ridge.reset_index(inplace=True, drop=True)
    
    for var in all_vars: 
        if var not in df_saturation_ridge.columns:
            df_saturation_ridge[var] = list(df_prophet[var])
            
    df_saturation_ridge[dep_var] = list(df_window[dep_var])
    df_saturation_ridge[date_var] = list(df_window[date_var])
    df_saturation_ridge.fillna(0, inplace=True)
    
    ridge_coefs = [value['coef'] for key, value in summary_dict.items() if key != 'intercept']
    if use_intercept:
        ridge_intercept = summary_dict['intercept']['coef']
    else:
        ridge_intercept = 0
    
    ridge_result, ridge_model = ridge(df_saturation_ridge, all_vars, dep_var, lambda_value=lambda_value, size=ridge_size, positive=ridge_positive, random_state=ridge_random_state, coeffs=ridge_coefs, intercept=ridge_intercept, context_vars=context)
    all_vars_use_intercept = all_vars.copy()
    
    if use_intercept:
        ridge_result['intercept'] = ridge_intercept
        all_vars_use_intercept.append('intercept')
    
    df_alldecomp_matrix = apply_decomp_transformation(ridge_result, all_vars_use_intercept, summary_dict, date_var)
    
    df_alldecomp_matrix[dep_var] = list(ridge_result[dep_var])
    df_alldecomp_matrix['prediction'] = list(ridge_result['prediction'])

    if type_of_use == 'refresh':
        # Filter the length of the data for errors to be calculated on Refresh Window only
        rsq=get_rsq_v2(ridge_result[dep_var].tail(json_model['InputCollect']['refresh_steps'][0]), ridge_result['prediction'].tail(json_model['InputCollect']['refresh_steps'][0]))
        nrmse=get_nrmse_v2(ridge_result[dep_var].tail(json_model['InputCollect']['refresh_steps'][0]), ridge_result['prediction'].tail(json_model['InputCollect']['refresh_steps'][0]))
        rssd=get_rssd_v2(df, df_alldecomp_matrix, paid_media, date_var, dep_var, is_refresh=True)

        if 'calibration_input' in json_model['InputCollect'] and json_model['InputCollect']['calibration_input']:
            calibration_input = json_model['InputCollect']['calibration_input']
            calibration_errors = []

            for elem in calibration_input:
                liftStartDate = elem['liftStartDate'].replace("as.Date(", "").strip()
                liftEndDate = elem['liftEndDate'].replace("as.Date(", "").strip()
                lift_response = float(elem['liftAbs'])

                df_alldecomp_channel = df_alldecomp_matrix.loc[(df_alldecomp_matrix[date_var] >= liftStartDate) & (df_alldecomp_matrix[date_var] <= liftEndDate)][elem['channel']]
                response_channel = df_alldecomp_channel.sum()

                calibration_error_channel = get_calibration_error(response_channel, lift_response)

                calibration_errors.append(calibration_error_channel)

            mape = np.nanmean(calibration_errors)
        else:
            mape=None
        
        return ridge_result, ridge_model, summary_dict, df_saturation_ridge, lambda_value, ridge_intercept, rsq, nrmse, rssd, mape
    
    return ridge_model, ridge_result, df_alldecomp_matrix, df_adstock, df_saturation, summary_dict

def prophet_cassandra(df, df_holidays, date_var, dep_var, prophet_vars, window_start='', window_end='',
        national_holidays_abbreviation='IT', future_dataframe_periods=28, freq='D', seasonality_mode='additive', is_predict_future=False, is_percentage_result=True):

    """
    Applies the Prophet forecasting model to a dataset, incorporating specified seasonality factors and national holidays. 
    It optionally predicts future values based on the model and returns a dataframe with the forecasted values and 
    specified seasonalities.

    Parameters:
    - df (DataFrame): The main dataset containing historical data for the dependent variable and dates.
    - df_holidays (DataFrame): A dataset containing holiday dates for the specified national holidays.
    - date_var (str): The column name in `df` representing the date.
    - dep_var (str): The column name in `df` representing the dependent variable to forecast.
    - prophet_vars (list): A list of variables indicating which seasonalities to include in the model ('trend', 'holiday', 'weekday', 'season', 'monthly').
    - window_start (str, optional): The start date for the prediction window (inclusive). Defaults to an empty string, indicating no start limit.
    - window_end (str, optional): The end date for the prediction window (inclusive). Defaults to an empty string, indicating no end limit.
    - national_holidays_abbreviation (str): The country code for which national holidays should be considered. Use '-' to ignore holidays.
    - future_dataframe_periods (int): The number of periods to forecast into the future.
    - freq (str): The frequency of the data recording (e.g., 'D' for daily).
    - seasonality_mode (str): The type of seasonality ('additive' or 'multiplicative').
    - is_predict_future (bool): Flag indicating whether to predict future values beyond the historical data.
    - is_percentage_result (bool): Flag indicating whether seasonal effects should be returned as percentages (relative change) or absolute values.

    Returns:
    - DataFrame: A dataframe containing the original date column and additional columns for each requested seasonality and trend, 
                 as well as the forecasted values for the dependent variable. If `is_predict_future` is True, the dataframe will 
                 be limited to the specified prediction window; otherwise, it merges the forecasted values with the original dataset.

    Note:
    - The function assumes the presence of `Prophet` from the `prophet` package and requires prior installation of this package.
    - This function is specifically designed for use within the Cassandra platform but can be adapted for other purposes.
    """
    
    if 'trend' in prophet_vars:
        trend_seasonality=True
    else:
        trend_seasonality=False
    if 'holiday' in prophet_vars:
        holiday_seasonality=True
    else:
        holiday_seasonality=False
    if 'weekday' in prophet_vars:
        weekday_seasonality=True
    else:
        weekday_seasonality=False
    if 'season' in prophet_vars:
        season_seasonality=True
    else:
        season_seasonality=False
    if 'monthly' in prophet_vars:
        monthly_seasonality=True
    else:
        monthly_seasonality=False
    
    # Create a DF with the only two columns for Prophet
    prophet_df = df[[date_var, dep_var]]

    # Rename the columns for Prophet
    prophet_df = prophet_df.rename(columns={date_var: 'ds', dep_var: 'y'})

    if national_holidays_abbreviation != '-':
        # Select the Holidays according to the country that interests me
        condition = (df_holidays['country'] == national_holidays_abbreviation)
        
        holidays = df_holidays.loc[condition, ['ds', 'holiday']]

        # Instance and fit Prophet
        prophet_m = Prophet(weekly_seasonality=weekday_seasonality, yearly_seasonality=season_seasonality,
                        daily_seasonality=False, holidays=holidays, seasonality_mode=seasonality_mode)
    else:
        # Instance and fit Prophet
        prophet_m = Prophet(weekly_seasonality=weekday_seasonality, yearly_seasonality=season_seasonality,
                        daily_seasonality=False, seasonality_mode=seasonality_mode)

    if monthly_seasonality:
        prophet_m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
    prophet_m.fit(prophet_df)
    
    future = prophet_m.make_future_dataframe(periods=int(future_dataframe_periods), freq=freq)

    forecast = prophet_m.predict(future)
    
    new_forecast = forecast[['ds', 'yhat', 'trend', 'additive_terms', 'multiplicative_terms']].copy()

    if 'yearly' in forecast:
        if is_percentage_result:
            new_forecast['season'] = forecast['yearly'].copy()
        else:
            new_forecast['season'] = forecast.apply(
                lambda row: row['trend'] * (row['yearly']),
                axis=1
            ) 
            
    if 'monthly' in forecast:
        if is_percentage_result:
            new_forecast['monthly'] = forecast['monthly'].copy()
        else:
            new_forecast['monthly'] = forecast.apply(
                lambda row: row['trend'] * row['monthly'],
                axis=1
            ) 
        
    if 'weekly' in forecast:
        if is_percentage_result:
            new_forecast['weekday'] = forecast['weekly'].copy()
        else:
            new_forecast['weekday'] = forecast.apply(
                lambda row: row['trend'] * row['weekly'],
                axis=1
            )
            
    if 'holidays' in forecast:
        if is_percentage_result:
            new_forecast['holidays'] = forecast['holidays'].copy()
        else:
            new_forecast['holidays'] = forecast.apply(
                lambda row: row['trend'] * row['holidays'],
                axis=1
            )

    sub_prophet_df = new_forecast[['ds']].copy()

    if trend_seasonality:
        sub_prophet_df['trend'] = new_forecast['trend']
    if holiday_seasonality:
        sub_prophet_df['holiday'] = new_forecast['holidays']
    if 'season' in new_forecast:
        sub_prophet_df['season'] = new_forecast['season']
    if 'weekday' in new_forecast:
        sub_prophet_df['weekday'] = new_forecast['weekday']
    if 'monthly' in new_forecast:
        sub_prophet_df['monthly'] = new_forecast['monthly']

    sub_prophet_df = sub_prophet_df.rename(columns={'ds': date_var})

    df[date_var] = pd.to_datetime(df[date_var])
    sub_prophet_df[date_var] = pd.to_datetime(sub_prophet_df[date_var]) 
    
    if is_predict_future:
        if window_start and window_end:
            sub_prophet_df_window = sub_prophet_df.loc[(sub_prophet_df[date_var] >= window_start) & (sub_prophet_df[date_var] <= window_end)]
        elif window_start and not window_end:
            sub_prophet_df_window = sub_prophet_df.loc[sub_prophet_df[date_var] >= window_start]
        elif not window_start and window_end:
            sub_prophet_df_window = sub_prophet_df.loc[sub_prophet_df[date_var] <= window_end]
        else:
            sub_prophet_df_window = sub_prophet_df
        
        return sub_prophet_df_window

    # Step 1: Identify Overlapping Columns
    overlapping_columns = set(df.columns).intersection(sub_prophet_df.columns)
    overlapping_columns.remove(date_var)  # Exclude the merge column

    full_df = pd.merge(df, sub_prophet_df, how='inner', on=date_var)

    # Step 3: Keep Columns from sub_prophet_df
    for col in overlapping_columns:
        full_df[col] = full_df[col + '_y']
        full_df.drop([col + '_x', col + '_y'], axis=1, inplace=True)

    if window_start and window_end:
        df_window = full_df.loc[(full_df[date_var] >= window_start) & (full_df[date_var] <= window_end)]
    elif window_start and not window_end:
        df_window = full_df.loc[full_df[date_var] >= window_start]
    elif not window_start and window_end:
        df_window = full_df.loc[full_df[date_var] <= window_end]
    else:
        df_window = full_df
    
    return df_window

def adstock(x, shape, scale, type="pdf"):
    windlen = len(x)
    x_decayed = np.zeros(windlen)
    thetaVecCum = np.zeros(windlen)
    
    if windlen <= 1:
        x_decayed = np.array(x)
        thetaVecCum = np.zeros(windlen)
        x_imme = None
    
    else:
        x_bin = np.arange(1, windlen + 1)
        scale_trans = np.round(np.quantile(np.arange(1, windlen + 1), scale), 0)
        '''if type.lower() == "cdf":
            thetaVec = np.concatenate(([1], 1 - weibull_min.cdf(x_bin[:-1], shape, scale=scaleTrans)))
            thetaVecCum = np.cumprod(thetaVec)'''
            
        '''if type.lower() == "pdf":'''
        thetaVecCum = normalize(weibull_min.pdf(x_bin, c=shape, scale=scale_trans))
        
        for i, x_val in enumerate(x):
            x_vec = np.concatenate((np.zeros(i), np.repeat(x_val, windlen - i)))
            thetaVecCumLag = np.concatenate((np.zeros(i), thetaVecCum[:windlen-i]))
            x_decayed += x_vec * thetaVecCumLag
        
        x_imme = np.diag(np.outer(x, thetaVecCum))
            
        '''x_decayed = [decay(x_val, x_pos, theta_vec_cum, windlen) for x_val, x_pos in zip(x, x_bin[:len(x)])]
        x_imme = np.diag(x_decayed)
        x_decayed = np.sum(x_decayed, axis=0)'''
    
    inflation_total = np.sum(x_decayed) / np.sum(x) if np.sum(x) != 0 else 0
    
    return {"x": x, "x_decayed": x_decayed, "theta_vec_cum": thetaVecCum, "inflation_total":inflation_total, "x_imme": x_imme}

def normalize(x):
    """Normalize the input array."""
    range_x = np.max(x) - np.min(x)
    if range_x == 0:
        return np.concatenate(([1], np.zeros(len(x) - 1)))
    else:
        return (x - np.min(x)) / range_x

def decay(x_val, x_pos, theta_vec_cum, windlen):
    x_vec = np.concatenate([np.zeros(x_pos - 1), np.full(windlen - x_pos + 1, x_val)])
    theta_vec_cum_lag = list(pd.Series(theta_vec_cum.copy()).shift(periods=x_pos-1, fill_value=0))
    x_prod = x_vec * theta_vec_cum_lag
    return x_prod

def saturation_hill(x, alpha, gamma, x_marginal=None):
    inflexion = (np.min(x) * (1 - gamma)) + (np.max(x) * gamma)# linear interpolation by dot product
    if x_marginal is None:
        x_scurve = x**alpha / (x**alpha + inflexion**alpha) # plot(x_scurve) summary(x_scurve)
    else:
        x_scurve = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
    return x_scurve

def ridge(df, all_vars, dep_var, lambda_value=0, size=0.2, positive=False, random_state=42, coeffs=[], intercept=0, fit_intercept=True, context_vars=[]):
        
        if context_vars:
            has_string = {col: df[col].apply(type).eq(str).any() for col in context_vars}

            for col, contains_str in has_string.items():
                if contains_str:
                    unique_strings = df.loc[df[col].apply(type).eq(str), col].unique()
                    mapping_dict = {unique_str: idx + 1 for idx, unique_str in enumerate(unique_strings)}
                    df[col] = df[col].replace(mapping_dict)
        
        X = df[all_vars]
        y = df[dep_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=random_state)

        model = Ridge(alpha=lambda_value, fit_intercept=fit_intercept, positive=positive)
        model.intercept_ = intercept
        model.coef_ = np.array(coeffs)
        model.fit(X_train, y_train)
        
        model.intercept_ = intercept
        model.coef_ = np.array(coeffs)

        # Ask the model to predict on X_test without having Y_test
        # This will give you exact predicted values

        # We can use our NRMSE and MAPE functions as well

        # Create new DF not to edit the original one
        result = df.copy()

        # Create a new column with predicted values
        result['prediction'] = model.predict(X)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        metrics_values = {}

        return result, model

def get_rsq_v2(y_true, y_pred):
    if len(y_true) == 1 and len(y_pred) == 1:
        difference = abs(y_true[0] - y_pred[0])
        value = max(0, min(100, 100 - difference * 10))
    else:
        corr_matrix = np.corrcoef(list(y_true), list(y_pred))
        corr = corr_matrix[0,1]
        value = corr**2
        value = max(0, min(100, value))
    
    return value

# Se si desidera confrontare errori tra serie con diversi range, questa versione potrebbe essere più appropriata.
def get_nrmse_v2(y_true, y_pred):
    if len(y_true) == 1 and len(y_pred) == 1:
        if y_true[0] != 0:
            difference = abs(y_true[0] - y_pred[0]) / y_true[0]
            value = min(100, difference * 100)
        else: 
            value = 0

    else:
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        range_y = np.max(y_true) - np.min(y_true)

        value = rmse / range_y

        value = max(0, min(100, value))
    
    return value

def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Avoid division by zero and calculate MAPE
    mask = y_true != 0
    mape = (abs((y_true - y_pred) / y_true)[mask]).mean()
        
    return mape

def get_rssd_v2(df, df_alldecomp, media, date_var, dep_var, is_refresh=False):
    
    if is_refresh:

        start_date = min(df_alldecomp[date_var])
        end_date = max(df_alldecomp[date_var])

        df_media_spend = df.copy()[media]
        df_media_spend[date_var] = df[date_var]
        df_media_spend = df_media_spend.loc[(df_media_spend[date_var] >= start_date) & (df_media_spend[date_var] <= end_date)]
        
        share_value = {}
        df_spend = df_media_spend.copy()
        
        df_waterfall = df_alldecomp.copy()
        df_waterfall.drop([date_var, dep_var, 'prediction'], axis=1, inplace=True)
        
        df_waterfall.drop_duplicates(inplace=True)
                            
        decomp_channels = list(df_waterfall.columns)
        if "trend" in decomp_channels and "intercept" in decomp_channels:
            decomp_channels.remove("intercept")
        df_spend.drop(date_var, axis=1, inplace=True)

    else:

        start_date = min(df_alldecomp['ds'])
        end_date = max(df_alldecomp['ds'])

        df_media_spend = df.copy()[media]
        df_media_spend['ds'] = df[date_var]
        df_media_spend = df_media_spend.loc[(df_media_spend['ds'] >= start_date) & (df_media_spend['ds'] <= end_date)]

        share_value = {}
        df_spend = df_media_spend.copy()

        df_waterfall = df_alldecomp.copy()
        df_waterfall.drop(['ds', 'dep_var', 'depVarHat', 'solID'], axis=1, inplace=True)

        columns_to_drop = ['Unnamed: 0', 'refreshStatus', 'bestModRF', 'cluster', 'top_sol']
    
        for column in columns_to_drop:
            if column in df_waterfall.columns:
                df_waterfall.drop(column, axis=1, inplace=True)

        df_waterfall.drop_duplicates(inplace=True)

        decomp_channels = list(df_waterfall.columns)
        if "trend" in decomp_channels and "intercept" in decomp_channels:
            decomp_channels.remove("intercept")
        df_spend.drop('ds', axis=1, inplace=True)

    decomp_decomposition_spend = []
    decomp_decomposition_response = []

    for channels in list(df_spend.columns):
        decomp_decomposition_spend.append(round(sum(df_spend[channels]), 2))
    decomp_ch_dec_spend_dirty = dict(zip(list(df_spend.columns), decomp_decomposition_spend))

    for channels in decomp_channels:
        decomp_decomposition_response.append(round(sum(df_waterfall[channels]), 2))
    decomp_ch_dec_response_dirty = dict(zip(decomp_channels, decomp_decomposition_response))

    share_value['channels']=[]
    share_value['spend'] = []
    share_value['effect'] = []

    for ch in decomp_channels:
        if ch in media:
            ch_spend = 0 
            ch_response = 0  
            for key, value in decomp_ch_dec_response_dirty.items():
                if key == ch:
                    ch_response = value

            for key, value in decomp_ch_dec_spend_dirty.items():
                if key == ch:  
                    if not isinstance(check_if_exist_value(value), str):
                        ch_spend = value
                        
            share_value['channels'].append(ch)
            share_value['spend'].append(ch_spend)
            share_value['effect'].append(ch_response)
    
    all_spend = sum(share_value['spend'])
    all_effect = sum(share_value['effect'])

    channels_dict = {}
    channels_dict['channels']=[]
    channels_dict['spend'] = []
    channels_dict['effect'] = []

    for ch in decomp_channels:
        if ch in media:
            ch_perc_spend = 0 
            ch_perc_response = 0
            for key, value in decomp_ch_dec_response_dirty.items():
                if key == ch:
                    if not isinstance(check_if_exist_value(value), str):
                        ch_perc_response = round((value / all_effect), 2)

            for key, value in decomp_ch_dec_spend_dirty.items():
                if key == ch:  
                    if not isinstance(check_if_exist_value(value), str):
                        ch_perc_spend = round((value / all_spend), 2)
        
            channels_dict['channels'].append(ch)
            channels_dict['spend'].append(ch_perc_spend)
            channels_dict['effect'].append(ch_perc_response)

    value = np.sqrt(np.sum((np.array(channels_dict['effect']) - np.array(channels_dict['spend'])) ** 2))

    value = max(0, min(100, value))
    
    return value

def get_calibration_error(effect_share, input_calibration):
    value = abs(round((effect_share - input_calibration) / input_calibration, 2))

    if value > 100:
        value = 100
    elif value < 0:
        value = 0
    
    return value

def create_error_metrics(json_model, df, df_alldecomp, key_export_model = 'ExportedModel'):
    
    all_sol_id = json_model[key_export_model]['select_model'] + json_model['InputCollect']['refreshSourceID'] if 'refreshSourceID' in json_model['InputCollect'] else json_model[key_export_model]['select_model']
    df_alldecomp_new = df_alldecomp.loc[(df_alldecomp['solID'].isin(all_sol_id))]
    all_media = json_model['InputCollect']['all_media']
    paid_media = json_model['InputCollect']['paid_media_spends']
    date_var = json_model['InputCollect']['date_var'][0]
    dep_var = json_model['InputCollect']['dep_var'][0]

    columns_to_drop = ['Unnamed: 0', 'refreshStatus', 'bestModRF', 'cluster', 'top_sol']
    
    for column in columns_to_drop:
        if column in df_alldecomp_new.columns:
            df_alldecomp_new.drop(column, axis=1, inplace=True)
        
    #df_alldecomp_new = df_alldecomp_new[['ds', 'dep_var', 'depVarHat']]
    df_alldecomp_new.drop_duplicates(inplace=True)
    is_jsons_model_changed = False
    is_train_test_validation = False

    if 'train_size' in json_model[key_export_model]['hyper_values'] and json_model[key_export_model]['hyper_values']['train_size'][0] < 1:
        is_train_test_validation = True

        train_size = json_model[key_export_model]['hyper_values']['train_size'][0]
        test_size, validation_size = (1 - train_size) / 2, (1 - train_size) / 2

        total_rows = len(df_alldecomp_new)
        
        train_split = int(total_rows * train_size)
        validation_split = train_split + int(total_rows * validation_size)
        test_split = validation_split + int(total_rows * test_size)
        remaining_rows = total_rows - test_split

        if remaining_rows == 1:
            test_split += 1 
        else:
            additional_validation_rows = remaining_rows // 2
            additional_test_rows = remaining_rows - additional_validation_rows

            validation_split += additional_validation_rows
            test_split += additional_test_rows + additional_validation_rows 

        # Split the DataFrame
        train_df = df_alldecomp_new.iloc[:train_split]
        validation_df = df_alldecomp_new.iloc[train_split:validation_split]
        test_df = df_alldecomp_new.iloc[validation_split:test_split]

        if 'rsq_train_cassandra' not in json_model[key_export_model]['errors'][0]:
            is_jsons_model_changed = True

            rsq_train_cassandra = get_rsq_v2(train_df['dep_var'], train_df['depVarHat'])

            if rsq_train_cassandra > 1:
                rsq_train_cassandra = 1
            elif rsq_train_cassandra < 0:
                rsq_train_cassandra = 0

            json_model[key_export_model]['errors'][0]['rsq_train_cassandra'] = rsq_train_cassandra

        else:
            rsq_train_cassandra = json_model[key_export_model]['errors'][0]['rsq_train_cassandra']
        
        if 'rsq_test_cassandra' not in json_model[key_export_model]['errors'][0]:
            is_jsons_model_changed = True

            rsq_test_cassandra = get_rsq_v2(test_df['dep_var'], test_df['depVarHat'])

            if rsq_test_cassandra > 1:
                rsq_test_cassandra = 1
            elif rsq_test_cassandra < 0:
                rsq_test_cassandra = 0

            json_model[key_export_model]['errors'][0]['rsq_test_cassandra'] = rsq_test_cassandra

        else:
            rsq_test_cassandra = json_model[key_export_model]['errors'][0]['rsq_test_cassandra']
        
        if 'rsq_validation_cassandra' not in json_model[key_export_model]['errors'][0]:
            is_jsons_model_changed = True

            rsq_validation_cassandra = get_rsq_v2(validation_df['dep_var'], validation_df['depVarHat'])

            if rsq_validation_cassandra > 1:
                rsq_validation_cassandra = 1
            elif rsq_validation_cassandra < 0:
                rsq_validation_cassandra = 0

            json_model[key_export_model]['errors'][0]['rsq_validation_cassandra'] = rsq_validation_cassandra

        else:
            rsq_validation_cassandra = json_model[key_export_model]['errors'][0]['rsq_validation_cassandra']
        
        if 'nrmse_train_cassandra' not in json_model[key_export_model]['errors'][0]:
            is_jsons_model_changed = True
            nrmse_train_cassandra = get_nrmse_v2(train_df['dep_var'], train_df['depVarHat'])

            if nrmse_train_cassandra > 1:
                nrmse_train_cassandra = 1
            elif nrmse_train_cassandra < 0:
                nrmse_train_cassandra = 0

            json_model[key_export_model]['errors'][0]['nrmse_train_cassandra'] = nrmse_train_cassandra

        else:
            nrmse_train_cassandra = json_model[key_export_model]['errors'][0]['nrmse_train_cassandra']
        
        if 'nrmse_test_cassandra' not in json_model[key_export_model]['errors'][0]:
            is_jsons_model_changed = True
            nrmse_test_cassandra = get_nrmse_v2(test_df['dep_var'], test_df['depVarHat'])

            if nrmse_test_cassandra > 1:
                nrmse_test_cassandra = 1
            elif nrmse_test_cassandra < 0:
                nrmse_test_cassandra = 0

            json_model[key_export_model]['errors'][0]['nrmse_test_cassandra'] = nrmse_test_cassandra

        else:
            nrmse_test_cassandra = json_model[key_export_model]['errors'][0]['nrmse_test_cassandra']
        
        if 'nrmse_validation_cassandra' not in json_model[key_export_model]['errors'][0]:
            is_jsons_model_changed = True

            nrmse_validation_cassandra = get_nrmse_v2(validation_df['dep_var'], validation_df['depVarHat'])

            if nrmse_validation_cassandra > 1:
                nrmse_validation_cassandra = 1
            elif nrmse_validation_cassandra < 0:
                nrmse_validation_cassandra = 0

            json_model[key_export_model]['errors'][0]['nrmse_validation_cassandra'] = nrmse_validation_cassandra

        else:
            nrmse_validation_cassandra = json_model[key_export_model]['errors'][0]['nrmse_validation_cassandra']
        
    if 'rsq_cassandra' not in json_model[key_export_model]['errors'][0]:
        is_jsons_model_changed = True     

        rsq_cassandra = get_rsq_v2(df_alldecomp_new['dep_var'], df_alldecomp_new['depVarHat'])

        if rsq_cassandra > 1:
            rsq_cassandra = 1
        elif rsq_cassandra < 0:
            rsq_cassandra = 0

        json_model[key_export_model]['errors'][0]['rsq_cassandra'] = rsq_cassandra

    else:
        rsq_cassandra = json_model[key_export_model]['errors'][0]['rsq_cassandra']
    
    if 'nrmse_cassandra' not in json_model[key_export_model]['errors'][0]:
        is_jsons_model_changed = True
        nrmse_cassandra = get_nrmse_v2(df_alldecomp_new['dep_var'], df_alldecomp_new['depVarHat'])

        if nrmse_cassandra > 1:
            nrmse_cassandra = 1
        elif nrmse_cassandra < 0:
            nrmse_cassandra = 0

        json_model[key_export_model]['errors'][0]['nrmse_cassandra'] = nrmse_cassandra

    else:
        nrmse_cassandra = json_model[key_export_model]['errors'][0]['nrmse_cassandra']
        
    if 'mape_cassandra' not in json_model[key_export_model]['errors'][0]:
        is_jsons_model_changed = True
        mape_cassandra = get_mape(df_alldecomp_new['dep_var'], df_alldecomp_new['depVarHat'])

        if mape_cassandra > 1:
            mape_cassandra = 1
        elif mape_cassandra < 0:
            mape_cassandra = 0

        json_model[key_export_model]['errors'][0]['mape_cassandra'] = mape_cassandra

    else:
        mape_cassandra = json_model[key_export_model]['errors'][0]['mape_cassandra']
    
    if 'rssd_cassandra' not in json_model[key_export_model]['errors'][0]:
        is_jsons_model_changed = True
        coefs = []
        for media in all_media:
            for summary in json_model[key_export_model]['summary']:
                if summary['variable'] == media:
                    coefs.append(summary['coef'])
                    break

        rssd_cassandra = get_rssd_v2(df, df_alldecomp_new, paid_media, date_var, dep_var)
        if rssd_cassandra > 1:
            rssd_cassandra = 1
        elif rssd_cassandra < 0:
            rssd_cassandra = 0

        json_model[key_export_model]['errors'][0]['rssd_cassandra'] = rssd_cassandra
    else:
        rssd_cassandra = json_model[key_export_model]['errors'][0]['rssd_cassandra']

    return json_model, is_jsons_model_changed, is_train_test_validation

def check_if_exist_value(elem):
    if pd.isna(elem) or math.isinf(elem) or elem == 0:
        elem = ' - '
    
    return elem

def create_confidence_interval(json_model, cluster_dict, bootstrap_df, share_spend, dep_var_type='O', key_export_model = 'ExportedModel'):
    
    paid_media = json_model['InputCollect']['paid_media_spends']
    target_solID = json_model[key_export_model]["select_model"][0]
    is_jsons_model_changed = False
    exist_ci_low_cassandra = True

    # Trovare il cluster corrispondente al solID
    target_cluster = None
    for cluster, solIDs in cluster_dict.items():
        if target_solID in solIDs:
            target_cluster = cluster
            break

    for elem in json_model[key_export_model]['summary']:
        if elem['variable'] == paid_media[0]:
            if 'ci_low_cassandra' not in elem.keys():
                exist_ci_low_cassandra = False
                break

    if not exist_ci_low_cassandra:
        is_jsons_model_changed = True
        for model_summary in json_model[key_export_model]["summary"]:
            if model_summary["variable"] in paid_media:
                if not bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['boot_mean'].empty:
                    model_summary["boot_mean_cassandra"] = bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['boot_mean'].iloc[0]
                else:
                    model_summary["boot_mean_cassandra"] = 0
                
                if not bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_up'].empty and not bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_low'].empty:
                    ci_up_cassandra = bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_up'].iloc[0]
                    ci_low_cassandra = bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_low'].iloc[0]
                        
                    ci_percentage = 1 - share_spend[model_summary["variable"]]
                    ci_width = model_summary["boot_mean_cassandra"] * ci_percentage

                    if ci_up_cassandra == ci_low_cassandra:
                        model_summary["ci_up_cassandra"] = ci_up_cassandra + ci_width
                        model_summary["ci_low_cassandra"] = ci_low_cassandra - ci_width
                    else:
                        model_summary["ci_up_cassandra"] = ci_up_cassandra if ci_up_cassandra != model_summary["boot_mean_cassandra"] else model_summary["boot_mean_cassandra"] + abs(ci_low_cassandra - model_summary["boot_mean_cassandra"])
                        model_summary["ci_low_cassandra"] = ci_low_cassandra if ci_low_cassandra != model_summary["boot_mean_cassandra"] else model_summary["boot_mean_cassandra"] - abs(ci_up_cassandra - model_summary["boot_mean_cassandra"])

                else:
                
                    if not bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_up'].empty:
                        ci_up_cassandra = bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_up'].iloc[0]
                        ci_percentage = 1 - share_spend[model_summary["variable"]]
                        ci_width = model_summary["boot_mean_cassandra"] * ci_percentage
                        
                        model_summary["ci_up_cassandra"] = ci_up_cassandra if ci_up_cassandra != model_summary["boot_mean_cassandra"] else ci_up_cassandra + ci_width
                    else:
                        model_summary["ci_up_cassandra"] = 0
                    
                    if not bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_low'].empty:
                        ci_low_cassandra = bootstrap_df[(bootstrap_df['cluster'] == target_cluster) & (bootstrap_df['rn'] == model_summary["variable"])]['ci_low'].iloc[0]
                        ci_percentage = 1 - share_spend[model_summary["variable"]]
                        ci_width = model_summary["boot_mean_cassandra"] * ci_percentage

                        model_summary["ci_low_cassandra"] = ci_low_cassandra if ci_low_cassandra != model_summary["boot_mean_cassandra"] else ci_low_cassandra - ci_width
                    else:
                        model_summary["ci_low_cassandra"] = 0

    return json_model, is_jsons_model_changed

def process_dashboard_data(mmm_dashboard: MmmDashboard):
    """
    Processes data from an MMM (Marketing Mix Modeling) dashboard, including reading and 
    analyzing the dataset and JSON model associated with the dashboard. It extracts and 
    formats various pieces of information critical for MMM analysis, such as paid media spends, 
    exposure variables, and modeling parameters.

    Args:
        mmm_dashboard (MmmDashboard): An instance of the MmmDashboard model, representing the 
                                      MMM dashboard to be processed.
        map_field (function): A function that maps media channel names to their corresponding 
                              field names in the dataset.

    Returns:
        dict: A dictionary containing key data elements extracted and processed from the 
              MMM dashboard. This includes the DataFrame loaded from the dashboard's dataset, 
              lists of paid media channels and their exposures, mapped media channels, 
              date-related variables, Prophet modeling variables (if any), and a summary of 
              the exported model. It also calculates the total actual spend based on actual 
              spend per day and the number of expected spend days.

              The keys in the returned dictionary are:
              - 'df': DataFrame containing the MMM dashboard dataset.
              - 'paid_media': List of paid media channels after excluding any removed channels.
              - 'paid_media_exposure': List of variables for paid media exposure.
              - 'mapped_paid_media': List of paid media channels mapped to their dataset field names.
              - 'date_var': The name of the date variable in the dataset.
              - 'dep_var': The name of the dependent variable for MMM analysis.
              - 'window_start': Start date for the analysis window.
              - 'window_end': End date for the analysis window.
              - 'refresh_added_start': Start date for when new data was added.
              - 'prophet_vars': Additional variables for Prophet modeling, if any.
              - 'prophet_country': The country setting for Prophet modeling.
              - 'type_aggregation': The type of data aggregation ('Daily', 'Weekly', 'Monthly').
              - 'freq': The frequency notation for time series analysis.
              - 'response': The type of response variable (e.g., 'Orders', 'Revenue').
              - 'cost_per_response': The cost metric associated with the response (e.g., 'CPO', 'ROI').
              - 'summary': Summary statistics or information from the exported model.
              - 'actual_spend': Actual spend data extracted from the dashboard.
              - 'total_actual_spend': Total actual spend calculated.
              - 'actual_expected_spend_days': The number of days for which actual spend is expected.
    """
    json_model = json.load(mmm_dashboard.json_file)

    paid_media = json_model['InputCollect']['paid_media_spends']
    if 'paidMediaSpendRemoved' in json_model['InputCollect']:
        paid_media_spend_removed = json_model['InputCollect']['paidMediaSpendRemoved']
        paid_media = [
            media for media in paid_media if media not in paid_media_spend_removed]
    else:
        paid_media_spend_removed = []
    
    if 'organic_vars' in json_model['InputCollect']:
        organic = json_model['InputCollect']['organic_vars']
        
        if 'organicVarsRemoved' in json_model['InputCollect']:
            organic_removed = json_model['InputCollect']['organicVarsRemoved']
            organic = [
                org for org in organic if org not in organic_removed]
        else:
            organic_removed = []
    else:
        organic = []
        
    if 'context_vars' in json_model['InputCollect']:
        context = json_model['InputCollect']['context_vars']
        
        if 'contextVarsRemoved' in json_model['InputCollect']:
            context_removed = json_model['InputCollect']['contextVarsRemoved']
            context = [
                ctx for ctx in context if ctx not in context_removed]
        else:
            context_removed = []
    else:
        context = []
        
    prophet_vars = json_model['InputCollect'].get('prophet_vars', [])

    all_ind_exposure_vars = json_model['InputCollect']['all_ind_vars'] + json_model['InputCollect']['exposure_vars']

    if prophet_vars:
        all_user_vars = [item for item in all_ind_exposure_vars if item not in prophet_vars] + json_model['InputCollect']['dep_var']
    else:
        all_user_vars = all_ind_exposure_vars + json_model['InputCollect']['dep_var']

    paid_media_exposure = json_model['InputCollect']['paid_media_vars']
    mapped_paid_media = [map_field(channel) for channel in paid_media]
    all_ind_vars = json_model['InputCollect']['all_ind_vars']
    all_media = json_model['InputCollect']['all_media']
    
    date_var = json_model['InputCollect']['date_var'][0]
    dep_var = json_model['InputCollect']['dep_var'][0]
    dep_var_type = json_model['InputCollect']['dep_var_type'][0]
    window_start = json_model['InputCollect']['window_start'][0]
    window_end = json_model['InputCollect']['window_end'][0]
    refresh_added_start = json_model['InputCollect']['refreshAddedStart'][0]
    
    prophet_country = json_model['InputCollect'].get(
        'prophet_country', ['-'])[0]

    interval_type = json_model['InputCollect']['intervalType'][0]
    if interval_type == 'day':
        type_aggregation = 'Daily'
        freq = 'D'
    elif interval_type == 'week':
        type_aggregation = 'Weekly'
        freq = 'W-MON'
    elif interval_type == 'month':
        type_aggregation = 'Monthly'
        freq = 'M'

    response_mapping = {
        'O': ('Orders', 'CPO'),
        'R': ('Revenue', 'ROI'),
        'N': ('New Customers', 'CAC'),
        'C': ('Conversions', 'CPA'),
    }
    response, cost_per_response = response_mapping.get(
        mmm_dashboard.response, ('Unknown', 'Unknown'))

    summary = json_model['ExportedModel']['summary']
    
    coefs = extract_coefficients_and_confidence_intervals(summary, paid_media)
    
    hypers, lambda_value = extract_hyperparameters(json_model['ExportedModel']['hyper_values'])
    
    select_model = json_model['ExportedModel']['select_model'][0]

    actual_spend, actual_expected_spend_days = take_actual_spend(
        mmm_dashboard_id=mmm_dashboard.id)
    
    df = pd.read_csv(mmm_dashboard.dataset)
    df.fillna(0, inplace=True)
    df = df.loc[df[date_var] <= window_end]
    
    df_alldecomp = pd.read_csv(mmm_dashboard.alldecomp)
    df_alldecomp = df_alldecomp.loc[df_alldecomp['solID'] == select_model].drop_duplicates(keep='first')
    
    json_model, is_jsons_model_changed, is_train_test_validation = create_error_metrics(
                json_model, df, df_alldecomp)
    
    error_metrics = {
        'rsq_cassandra': round(json_model['ExportedModel']['errors'][0]['rsq_cassandra']*100, 2),
        'nrmse_cassandra': round(json_model['ExportedModel']['errors'][0]['nrmse_cassandra']*100, 2),
        'mape_cassandra': round(json_model['ExportedModel']['errors'][0]['mape_cassandra']*100, 2),
        'rssd_cassandra': round(json_model['ExportedModel']['errors'][0]['rssd_cassandra']*100, 2)
    }

    return {
        'df': df,
        'df_alldecomp':df_alldecomp,
        'json_file':json_model,
        'paid_media': paid_media,
        'paid_media_exposure': paid_media_exposure,
        'mapped_paid_media': mapped_paid_media,
        'organic_vars': organic,
        'context_vars' : context,
        'all_ind_vars':all_ind_vars,
        'all_media':all_media,
        'all_user_vars':all_user_vars,
        'date_var': date_var,
        'dep_var': dep_var,
        'dep_var_type':dep_var_type,
        'window_start': window_start,
        'window_end': window_end,
        'refresh_added_start': refresh_added_start,
        'prophet_vars': prophet_vars,
        'prophet_country': prophet_country,
        'type_aggregation': type_aggregation,
        'freq': freq,
        'interval_type':interval_type,
        'response': response,
        'cost_per_response': cost_per_response,
        'summary': summary,
        'coefs':coefs,
        'hypers':hypers,
        'lambda':lambda_value,
        'actual_spend': actual_spend,
        'total_actual_spend': actual_spend * actual_expected_spend_days,
        'actual_expected_spend_days': actual_expected_spend_days,
        'select_model': select_model,
        'error_metrics': error_metrics,
        'is_jsons_model_changed': is_jsons_model_changed,
        'is_train_test_validation': is_train_test_validation,
        'is_created_by_cassandra': mmm_dashboard.is_created_by_cassandra
    }

class MmmDashboard(models.Model):
    TYPE_OF_RESPONSE = (
        ("O", "Orders"),
        ("C", "Conversions"),
        ("R", "Revenue"),
        ("N", "New Customers"),
    )

    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
    )

    previous_dashboard = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="refresh_dashboards",
    )

    model = models.CharField(max_length=100, null=True, blank=True)
    response = models.CharField(
        default="O", max_length=100, choices=TYPE_OF_RESPONSE, null=True, blank=True
    )
    sol_id = models.CharField(max_length=100, null=False, blank=True)

    # TO REMOVE
    alldecomp = models.FileField(null=True, blank=True)
    json_file = models.FileField(null=True, blank=True)
    dataset = models.FileField(null=True, blank=True)

    schema_field = models.JSONField(null=True, blank=True)

    budget_allocator_default = models.IntegerField(null=True, blank=True)
    budget_forecast_default = models.IntegerField(null=True, blank=True)
    channel_default = models.CharField(max_length=100, null=True, blank=True)

    is_created_by_cassandra = models.BooleanField(default=False, null=False)
    is_simplified = models.BooleanField(default=False, null=False)
    is_free_trial = models.BooleanField(default=False, null=False)
    is_first_view = models.BooleanField(default=True, null=False)
    is_merged = models.BooleanField(default=False, null=False)

    # Method to update the previous_dashboard field
    def update_previous_dashboard(self):
        try:
            # Read the JSON file and load it into a Python dictionary
            with self.json_file.open() as f:
                json_data = json.load(f)

            # Extract the refreshSourceID, assuming it's an array in the JSON under InputCollect
            refresh_source_ids = json_data.get("InputCollect", {}).get(
                "refreshSourceID", []
            )

            # Take the first element from the array
            if refresh_source_ids:
                first_refresh_source_id = refresh_source_ids[0]

                # Search for a start dashboard with this ID in its json_file['ExportedModel']['select_model'][0]
                previous_dashboard = MmmDashboard.objects.filter(
                    sol_id=first_refresh_source_id
                )

                if previous_dashboard.exists():
                    self.previous_dashboard = previous_dashboard.first()
                    self.save()

        except Exception as e:
            print(
                f"Error updating previous_dashboard {self.previous_dashboard_id}: {e}"
            )

    def save_file(self, file, type):
        type_to_field_extension = {
            "dataset": ("dataset", "csv"),
            "alldecomp": ("alldecomp", "csv"),
            "json_file": ("json_file", "json"),
        }

        if type not in type_to_field_extension:
            # Handle unsupported type
            print("Unsupported file type.")
            return

        field_name, extension = type_to_field_extension[type]

        if extension == "csv":
            content = file.to_csv(index=False)
        else:  # JSON
            content = json.dumps(file)

        content_file = ContentFile(content.encode("utf-8"))

        # Generate a file name based on unique attributes
        file_name = f"{type}_{self.id}.{extension}"

        # Save the file to the appropriate field
        getattr(self, field_name).save(file_name, content_file)
        self.save()

    def save_from_dict(self, data_dict):
        """
        Updates the DataAnalysis instance from a dictionary of parameters and optionally handles related DataFrame.

        Parameters:
        - data_dict (dict): Dictionary containing configurations, parameters, and analysis results.
        """
        for key, value in data_dict.items():
            if key in ["dataset", "alldecomp", "json_file"]:
                self.save_file(value, key)
            else:
                setattr(self, key, value)

        self.save()

    creation_date = models.DateTimeField(auto_now_add=True, null=True)
    order_with_respect_to = "creation_date"

    class Meta:
        ordering = ["id"]