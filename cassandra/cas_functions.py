import numpy as np
# from prophet import Prophet
from scipy.stats import weibull_min, norm, t
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
import pandas as pd
import numpy as np
# import nevergrad as ng
import random
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


def adstock(x, shape, scale, type="pdf"):
    verbose = False
    if verbose:
        print(x.name)
        print(len(x))
        print(shape)
        print(scale)

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
            thetaVecCumLag = np.concatenate((np.zeros(i), thetaVecCum[:windlen -i]))
            x_decayed += x_vec * thetaVecCumLag

        x_imme = np.diag(np.outer(x, thetaVecCum))

        '''x_decayed = [decay(x_val, x_pos, theta_vec_cum, windlen) for x_val, x_pos in zip(x, x_bin[:len(x)])]
        x_imme = np.diag(x_decayed)
        x_decayed = np.sum(x_decayed, axis=0)'''

    inflation_total = np.sum(x_decayed) / np.sum(x) if np.sum(x) != 0 else 0

    return {"x": x, "x_decayed": x_decayed, "theta_vec_cum": thetaVecCum, "inflation_total" :inflation_total, "x_imme": x_imme}



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


def normalize(x):
    """Normalize the input array."""
    range_x = np.max(x) - np.min(x)
    if range_x == 0:
        return np.concatenate(([1], np.zeros(len(x) - 1)))
    else:
        return (x - np.min(x)) / range_x


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


def saturation_hill(x, alpha, gamma, x_marginal=None):
    inflexion = (np.min(x) * (1 - gamma)) + (np.max(x) * gamma)# linear interpolation by dot product
    if x_marginal is None:
        x_scurve = x**alpha / (x**alpha + inflexion**alpha) # plot(x_scurve) summary(x_scurve)
    else:
        x_scurve = x_marginal**alpha / (x_marginal**alpha + inflexion**alpha)
    return x_scurve


def import_model(json_model, df, df_holidays, prophet_future_dataframe_periods=14,
                 prophet_seasonality_mode='additive', ridge_size=0.2, ridge_positive=True, ridge_random_state=42,
                 type_of_use='import'):
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
        prophet_freq = 'D'
    elif interval_type == 'week':
        prophet_freq = 'W'
    elif interval_type == 'month':
        prophet_freq = 'M'

    df_window = df.loc[(df[date_var] >= window_start) & (df[date_var] <= window_end)]
    df_window.reset_index(inplace=True, drop=True)

    ####################################################SECTION PROPHET####################################################

    df_copy = df.copy()

    df_prophet = prophet_cassandra(df_copy, df_holidays, date_var, dep_var, prophet_vars, window_start='',
                                   window_end='',
                                   national_holidays_abbreviation=prophet_country,
                                   future_dataframe_periods=prophet_future_dataframe_periods, freq=prophet_freq,
                                   seasonality_mode=prophet_seasonality_mode)

    df_prophet = df_prophet.loc[(df_prophet[date_var] >= window_start) & (df_prophet[date_var] <= window_end)]
    df_prophet.reset_index(inplace=True, drop=True)
    ####################################################SECTION COEFF AND HYPERPARAMETERS####################################################

    # take a coef
    coef_dict = extract_coefficients_and_confidence_intervals(json_model['ExportedModel']['summary'], paid_media)

    # take a hyperparameters
    hyper_dict, lambda_value = extract_hyperparameters(json_model['ExportedModel']['hyper_values'])

    summary_dict = create_summary_dictionary(all_vars, hyper_dict, coef_dict)

    use_intercept = True if summary_dict['intercept']['coef'] != 0 else False

    ####################################################SECTION CREATE ADSTOCK AND SATURATION DATASET####################################################

    df_adstock = apply_adstock_transformation(df, all_media, summary_dict, date_var)

    df_saturation, df_adstock_filtered = apply_saturation_transformation(df_adstock, all_media, summary_dict, date_var,
                                                                         window_start, window_end)

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

    ridge_result, ridge_model = ridge(df_saturation_ridge, all_vars, dep_var, lambda_value=lambda_value,
                                      size=ridge_size, positive=ridge_positive, random_state=ridge_random_state,
                                      coeffs=ridge_coefs, intercept=ridge_intercept, context_vars=context)
    all_vars_use_intercept = all_vars.copy()

    if use_intercept:
        ridge_result['intercept'] = ridge_intercept
        all_vars_use_intercept.append('intercept')

    df_alldecomp_matrix = apply_decomp_transformation(ridge_result, all_vars_use_intercept, summary_dict, date_var)

    df_alldecomp_matrix[dep_var] = list(ridge_result[dep_var])
    df_alldecomp_matrix['prediction'] = list(ridge_result['prediction'])

    if type_of_use == 'refresh':
        # Filter the length of the data for errors to be calculated on Refresh Window only
        rsq = get_rsq_v2(ridge_result[dep_var].tail(json_model['InputCollect']['refresh_steps'][0]),
                         ridge_result['prediction'].tail(json_model['InputCollect']['refresh_steps'][0]))
        nrmse = get_nrmse_v2(ridge_result[dep_var].tail(json_model['InputCollect']['refresh_steps'][0]),
                             ridge_result['prediction'].tail(json_model['InputCollect']['refresh_steps'][0]))
        rssd = get_rssd_v2(df, df_alldecomp_matrix, paid_media, date_var, dep_var, is_refresh=True)

        if 'calibration_input' in json_model['InputCollect'] and json_model['InputCollect']['calibration_input']:
            calibration_input = json_model['InputCollect']['calibration_input']
            calibration_errors = []

            for elem in calibration_input:
                liftStartDate = elem['liftStartDate'].replace("as.Date(", "").strip()
                liftEndDate = elem['liftEndDate'].replace("as.Date(", "").strip()
                lift_response = float(elem['liftAbs'])

                df_alldecomp_channel = df_alldecomp_matrix.loc[
                    (df_alldecomp_matrix[date_var] >= liftStartDate) & (df_alldecomp_matrix[date_var] <= liftEndDate)][
                    elem['channel']]
                response_channel = df_alldecomp_channel.sum()

                calibration_error_channel = get_calibration_error(response_channel, lift_response)

                calibration_errors.append(calibration_error_channel)

            mape = np.nanmean(calibration_errors)
        else:
            mape = None

        return ridge_result, ridge_model, summary_dict, df_saturation_ridge, lambda_value, ridge_intercept, rsq, nrmse, rssd, mape

    return ridge_model, ridge_result, df_alldecomp_matrix, df_adstock, df_saturation, summary_dict, df_prophet


def prophet_cassandra(df, df_holidays, date_var, dep_var, prophet_vars, window_start='', window_end='',
                      national_holidays_abbreviation='IT', future_dataframe_periods=28, freq='D',
                      seasonality_mode='additive', is_predict_future=False, is_percentage_result=True):
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
        trend_seasonality = True
    else:
        trend_seasonality = False
    if 'holiday' in prophet_vars:
        holiday_seasonality = True
    else:
        holiday_seasonality = False
    if 'weekday' in prophet_vars:
        weekday_seasonality = True
    else:
        weekday_seasonality = False
    if 'season' in prophet_vars:
        season_seasonality = True
    else:
        season_seasonality = False
    if 'monthly' in prophet_vars:
        monthly_seasonality = True
    else:
        monthly_seasonality = False

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
            sub_prophet_df_window = sub_prophet_df.loc[
                (sub_prophet_df[date_var] >= window_start) & (sub_prophet_df[date_var] <= window_end)]
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


def get_calibration_error(effect_share, input_calibration):
    value = abs(round((effect_share - input_calibration) / input_calibration, 2))

    if value > 100:
        value = 100
    elif value < 0:
        value = 0

    return value


def get_rsq_v2(y_true, y_pred):
    if len(y_true) == 1 and len(y_pred) == 1:
        difference = abs(y_true[0] - y_pred[0])
        value = max(0, min(100, 100 - difference * 10))
    else:
        corr_matrix = np.corrcoef(list(y_true), list(y_pred))
        corr = corr_matrix[0, 1]
        value = corr ** 2
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

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
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
        df_media_spend = df_media_spend.loc[
            (df_media_spend[date_var] >= start_date) & (df_media_spend[date_var] <= end_date)]

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

    share_value['channels'] = []
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
    channels_dict['channels'] = []
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


def ridge(df, all_vars, dep_var, lambda_value=0, size=0.2, positive=False, random_state=42, coeffs=[], intercept=0,
          fit_intercept=True, context_vars=[]):
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


def check_if_exist_value(elem):
    if pd.isna(elem) or math.isinf(elem) or elem == 0:
        elem = ' - '

    return elem


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