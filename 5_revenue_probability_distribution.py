
from utils import *
from cassandra.cas_functions import import_model
from cassandra.cas_functions import *
from cassandra.cas_allocator import *
from scipy.stats import gaussian_kde


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

common_date_list = list(common_dates)
all_vars = json_model['InputCollect']['all_ind_vars']
all_media =json_model['InputCollect']['all_media']
paid_media = json_model['InputCollect']['paid_media_vars']
paid_media_date = paid_media.copy()
paid_media_date.append(json_model['InputCollect']['date_var'][0])
date_var = json_model['InputCollect']['date_var'][0]
window_start = json_model['InputCollect']['window_start'][0]
window_end = json_model['InputCollect']['window_end'][0]
date_max = ridge_result['date_week'].max()

df_resample = df.copy()
df_resample = df_resample[df_resample[date_var].isin(common_date_list)]







import numpy as np
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d


def get_confidence_for_sample(df_saturation, decomp, media, date_var, sample_value):
    """
    Get confidence for a given sample of saturation.

    :param df_saturation: DataFrame containing saturation data
    :param decomp: DataFrame containing revenue decomposition
    :param media: The specific media column to analyze
    :param date_var: The date variable to use as the index
    :param sample_value: The sample saturation value for which to calculate confidence
    :return: Confidence for the given sample_value
    """
    df_saturation = df_saturation.copy()
    decomp = decomp.copy()

    # Preprocessing
    media_saturation = df_saturation[[date_var, media]].copy()
    media_revenue = decomp[[date_var, media]].copy()
    media_saturation.set_index(date_var, inplace=True)
    media_revenue.set_index(date_var, inplace=True)
    media_saturation['revenue'] = media_revenue[media]
    media_saturation.columns = ['saturation', 'revenue']
    media_saturation = media_saturation.sort_values(by='saturation')

    # Step 1: Calculate the local density using Gaussian Kernel Density Estimation (KDE)
    kde = gaussian_kde(media_saturation['saturation'], bw_method='scott')
    media_saturation['density'] = kde(media_saturation['saturation'])

    # Step 2: Calculate confidence based on density (higher density = higher confidence)
    media_saturation['confidence'] = (media_saturation['density'] - media_saturation['density'].min()) / \
                                     (media_saturation['density'].max() - media_saturation['density'].min())

    # Step 3: Find the closest saturation samples and interpolate if necessary
    saturation_values = media_saturation['saturation'].values
    confidence_values = media_saturation['confidence'].values

    if sample_value < saturation_values.min():
        # Below the range, return the confidence of the closest (smallest) value
        confidence = confidence_values[0]
    elif sample_value > saturation_values.max():
        # Above the range, return the confidence of the closest (largest) value
        confidence = confidence_values[-1]
    else:
        # Within the range, perform linear interpolation
        interp_func = interp1d(saturation_values, confidence_values, kind='linear')
        confidence = interp_func(sample_value)

    return confidence


def generate_sample_optimal_budget():
    exp_spend_unit_total = 18124
    init_spend_unit = [0.0, 496.955, 275.7, 0.0, 0.0, 540.9200000000001, 32.294999999999995, 947.98, 2014.315, 223.0,
                       0.0, 0.0]
    period_to_estimate = 4
    channel_constr_low_sorted = [0, 367.3598589405508, 214.83203998480366, 0, 0, 387.9317747252748, 23.420789163628577,
                                 801.7245691391942, 1547.317561793812, 168.19980933924222, 0, 0]
    channel_constr_up_sorted = [175.12832298136647, 626.5501410594492, 336.5679600151963, 0, 197.54807453416151,
                                693.9082252747254, 41.16921083637141, 1094.235430860806, 2481.3124382061883,
                                277.8001906607578, 377.5833333333333, 40.46204968944099]
    channelConstrMeanSorted = [(low + up) / 2 for low, up in zip(channel_constr_low_sorted, channel_constr_up_sorted)]

    paid_media_spends = [
        channelConstrMeanSorted[i] if spend < channel_constr_low_sorted[i] or spend > channel_constr_up_sorted[
            i] else spend for i, spend in enumerate(init_spend_unit)]

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

    budget_allocator_spend = budget_allocation(paid_media_spends, channel_constr_low_sorted, channel_constr_up_sorted,
                                               period_to_estimate, 700, 1e-10, eval_list, paid_media_spend_traspose,
                                               period_to_estimate, target_value, 'R')

    # new_dates = extend_date_range(df, date_var, period_to_estimate, 'Weekly', date_max)
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

    first_predicted_week = df_optim['date_week'].iloc[-period_to_estimate]

    return df_optim_saturation, df_optim_decomp, first_predicted_week



def generate_revenue_confidence_bounds():

    """ This function analyse the confidence bounds for the revenue prediction.
    The analysis is perfomed only on paid media channel and the focus is on the first optimal budget predicted row.
    """

    df_optim_saturation, df_optim_decomp, first_predicted_week = generate_sample_optimal_budget()
    df_optim_row = df_optim_decomp[df_optim_decomp[date_var] == first_predicted_week]
    df_optim_row_upper = df_optim_row.copy()
    df_optim_row_lower = df_optim_row.copy()
    M = 0.5

    verbose = False
    for media in paid_media:
        sample_df_optim = df_optim_saturation[df_optim_saturation[date_var] == first_predicted_week][media].values[0]
        confidence = get_confidence_for_sample(df_saturation, df_alldecomp_matrix, media, date_var, sample_df_optim)

        if verbose:
            print(f'Confidence for {media}: {np.round(confidence,3)}')

        df_optim_row_upper[media] = df_optim_row_upper[media] * (1 + M * (1 - confidence))
        df_optim_row_lower[media] = df_optim_row_lower[media] * (1 - M * (1 - confidence))


    total_revenue_row = df_optim_row[paid_media].sum(axis=1)
    total_revenue_row_upper = df_optim_row_upper[paid_media].sum(axis=1)
    total_revenue_row_lower = df_optim_row_lower[paid_media].sum(axis=1)


    print(f'Total revenue: {total_revenue_row.values[0]}')
    print(f'Total revenue upper bound: {total_revenue_row_upper.values[0]}')
    print(f'Total revenue lower bound: {total_revenue_row_lower.values[0]}')


if __name__ == '__main__':
    generate_revenue_confidence_bounds()



#get_confidence_for_sample(df_saturation, df_alldecomp_matrix, paid_media[5], date_var, 0.05)