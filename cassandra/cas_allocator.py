import datetime
import pandas as pd
import numpy as np
import warnings
import math
from datetime import datetime, timedelta
import nlopt

from .cas_functions import create_summary_dictionary, apply_adstock_transformation,\
                                apply_decomp_transformation, apply_saturation_transformation, adstock, \
                                saturation_hill


from .functions_date import calculate_allocation_expected_spend_days, adjust_date, \
                        calculate_start_date_based_on_aggregation, adjust_allocation_dates_for_aggregation, \
                        format_dates, adjust_dates_for_weekly_aggregation, adjust_dates_for_monthly_aggregation


def setup_optimal_info(budget_allocator, mmm_dashboard_config, hist_spend_unit, exp_pend_unit_total, init_spend_unit,
                       init_spend_unit_total, init_response_unit, init_response_unit_total):
    """
    Sets up the optimal information for budget allocation based on the given parameters.

    Args:
        budget_allocator (object): The budget allocator object.
        mmm_dashboard_config (dict): The configuration for the MMM (Marketing Mix Modeling) dashboard.
        hist_spend_unit (list): The historical spend units.
        exp_pend_unit_total (float): The expected pending unit total.
        init_spend_unit_total (float): The initial spend unit total.
        init_response_unit (list): The initial response units.
        init_response_unit_total (float): The initial response unit total.

    Returns:
        dict: The optimal information for budget allocation, including spend units, deltas, share, response units, ROI, and lifts.
    """

    number_of_paid_media = len(mmm_dashboard_config['paid_media'])

    allocation_dates = calculate_allocation_dates(budget_allocator, {
        'type_aggregation': mmm_dashboard_config['type_aggregation'],
        'window_end': mmm_dashboard_config['window_end']})

    periods_in_days, expected_spend_days_by_type_aggregation, periods_to_estimate, data_expected_spend = adjust_total(
        {'budget_start': allocation_dates['date_min'],
         'budget_end': allocation_dates['date_max'],
         'expected_spend': budget_allocator.expected_spend},
        mmm_dashboard_config['type_aggregation'])

    new_dates = extend_date_range(mmm_dashboard_config['df'], mmm_dashboard_config['date_var'], periods_to_estimate,
                                  mmm_dashboard_config['type_aggregation'], allocation_dates['date_max'])

    rollingWindow_new_dates = new_dates[new_dates.index(mmm_dashboard_config['window_start']) + 1:]
    rollingWindowLength = len(rollingWindow_new_dates)

    optimized_spend = find_optimized_spend(budget_allocator, mmm_dashboard_config, periods_to_estimate,
                                           exp_pend_unit_total, init_spend_unit, rollingWindowLength)

    summary_dict = create_summary_dictionary(mmm_dashboard_config['all_ind_vars'], mmm_dashboard_config['hypers'],
                                             mmm_dashboard_config['coefs'])

    df_optim = append_optimal_spend({'df': mmm_dashboard_config['df'],
                                     'paid_media': mmm_dashboard_config['paid_media'],
                                     'date_var': mmm_dashboard_config['date_var']}, optimized_spend,
                                    periods_to_estimate, allocation_dates['date_max'])

    df_optim[mmm_dashboard_config['date_var']] = new_dates

    df_optim_adstock = apply_adstock_transformation(df_optim, mmm_dashboard_config['paid_media'], summary_dict,
                                                    mmm_dashboard_config['date_var'])

    df_optim_saturation, df_optim_adstock_filtered = apply_saturation_transformation(df_optim_adstock,
                                                                                     mmm_dashboard_config['paid_media'],
                                                                                     summary_dict,
                                                                                     mmm_dashboard_config['date_var'],
                                                                                     mmm_dashboard_config[
                                                                                         'window_start'], new_dates[-1])

    df_optim_decomp = apply_decomp_transformation(df_optim_saturation, mmm_dashboard_config['paid_media'], summary_dict,
                                                  mmm_dashboard_config['date_var'])

    df_optim_decomp_reallocated_spend = df_optim_decomp.tail(periods_to_estimate)

    optm_spend_unit_delta = [(optm / hist) - 1 if hist != 0 else -1 for optm, hist in
                             zip(optimized_spend, hist_spend_unit)]
    optm_spend_unit_total = sum(optimized_spend)
    optm_spend_unit_total_delta = (optm_spend_unit_total / init_spend_unit_total) - 1
    optm_spend_share_unit = [(optm / sum(optimized_spend)) * 100 for optm in optimized_spend]

    optm_response_unit = df_optim_decomp_reallocated_spend[mmm_dashboard_config['paid_media']].mean().tolist()
    optm_response_unit_total = np.nansum(optm_response_unit)
    optm_roi_unit = [response / spend if int(spend) != 0 else 0 for response, spend in
                     zip(optm_response_unit, optimized_spend)]
    optm_response_unit_lift = [(optm / init) - 1 if not math.isnan(init) and int(init) != 0 else -1 for optm, init in
                               zip(optm_response_unit, init_response_unit)]
    optmResponseUnitTotalLift = (optm_response_unit_total / init_response_unit_total) - 1

    optimal_info = {
        'optmSpendUnit': optimized_spend,
        'optmSpendUnitDelta': optm_spend_unit_delta,
        'optmSpendUnitTotal': [optm_spend_unit_total] * number_of_paid_media,
        'optmSpendUnitTotalDelta': [optm_spend_unit_total_delta] * number_of_paid_media,
        'optmSpendShareUnit': optm_spend_share_unit,
        'optmResponseUnit': optm_response_unit,
        'optmResponseUnitTotal': [optm_response_unit_total] * number_of_paid_media,
        'optmRoiUnit': optm_roi_unit,
        'optmResponseUnitLift': optm_response_unit_lift,
        'optmResponseUnitTotalLift': [optmResponseUnitTotalLift] * number_of_paid_media,
    }
    return optimal_info


def calculate_allocation_dates(budget_allocator, mmm_dashboard_config, df_budget_allocator=pd.DataFrame()):
    """
    Calculate the allocation dates based on the given budget allocator, MMM dashboard configuration, and budget allocator DataFrame.

    Parameters:
    - budget_allocator: The budget allocator object.
    - mmm_dashboard_config: The MMM dashboard configuration dictionary.
    - df_budget_allocator: The budget allocator DataFrame (default: empty DataFrame).

    Returns:
    A tuple of formatted dates: (date_min, date_max, date_allocation_min, date_allocation_max).
    """

    if df_budget_allocator.empty:
        expected_spend_days = calculate_allocation_expected_spend_days(
            budget_allocator.expected_spend_days, {'type_aggregation': mmm_dashboard_config['type_aggregation']})

        date_max = adjust_date(datetime.strptime(mmm_dashboard_config['window_end'], '%Y-%m-%d'),
                               mmm_dashboard_config['type_aggregation'])
        date_min = calculate_start_date_based_on_aggregation(date_max, expected_spend_days,
                                                             mmm_dashboard_config['type_aggregation'])
    else:
        date_min = datetime.strptime(str(df_budget_allocator['date_min'].iloc[0]), '%Y-%m-%d')
        date_max = datetime.strptime(str(df_budget_allocator['date_max'].iloc[0]), '%Y-%m-%d')

    # Adjust allocation dates for aggregation
    dates = adjust_allocation_dates_for_aggregation(date_min, date_max, budget_allocator.expected_spend_days,
                                                    mmm_dashboard_config['type_aggregation'])

    # Format and return dates
    return format_dates(dates['date_min'], dates['date_max'], dates['date_allocation_min'],
                        dates['date_allocation_max'])


def extend_date_range(df, date_var, periods, type_aggregation, date_max):
    """
    Extend the date range of a DataFrame by adding additional dates.

    Args:
        df (pandas.DataFrame): The DataFrame containing the original date range.
        date_var (str): The name of the column in the DataFrame that contains the dates.
        periods (int): The number of additional periods to add to the date range.

    Returns:
        list: The extended date range as a list of strings in the format '%Y-%m-%d'.
    """

    df_tail = df.copy()
    df_tail = df_tail.loc[np.array(df_tail[date_var] <= date_max)] #  df_tail.loc[df_tail[date_var] <= date_max]
    last_date = datetime.strptime(df_tail[date_var].iloc[-1][0], '%Y-%m-%d')
    # additional_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, periods + 1)]
    if type_aggregation == 'Daily':
        additional_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, periods + 1)]
    elif type_aggregation == 'Weekly':
        additional_dates = [(last_date + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, periods + 1)]
    elif type_aggregation == 'Monthly':
        additional_dates = [(last_date + pd.DateOffset(months=i)).strftime('%Y-%m-%d') for i in range(1, periods + 1)]
    else:
        raise ValueError("Invalid type_aggregation. Expected 'Daily', 'Weekly', or 'Monthly'.")

    date_list = list(df_tail[date_var]) + additional_dates

    return date_list


def find_optimized_spend(budget_allocator, mmm_dashboard_config, periods_to_estimate, exp_spend_unit_total,
                         init_spend_unit, rollingWindowLength):
    """
    Finds the optimized spend based on the given budget allocator, MMM dashboard configuration, and expected spend unit total.

    Args:
        budget_allocator (BudgetAllocator): The budget allocator object.
        mmm_dashboard_config (dict): The MMM dashboard configuration dictionary.
        exp_spend_unit_total (float): The expected spend unit total.

    Returns:
        float: The optimized spend calculated by the budget allocator.

    Raises:
        None
    """
    summary_dict = create_summary_dictionary(mmm_dashboard_config['all_ind_vars'],
                                             mmm_dashboard_config['hypers'],
                                             mmm_dashboard_config['coefs'])

    channelConstrMeanSorted = [(low + up) / 2 for low, up in zip(budget_allocator.channel_constr_low_sorted,
                                                                 budget_allocator.channel_constr_up_sorted)]

    paid_media_spends = [channelConstrMeanSorted[i] if spend < budget_allocator.channel_constr_low_sorted[i] or spend >
                                                       budget_allocator.channel_constr_up_sorted[i] else spend for
                         i, spend in enumerate(init_spend_unit)]

    eval_list = {
        'coefsFiltered': [summary_dict[col]['coef'] for col in mmm_dashboard_config['paid_media']],
        'alphas': [summary_dict[col]['alphas'] for col in mmm_dashboard_config['paid_media']],
        'gammas': [summary_dict[col]['gammas'] for col in mmm_dashboard_config['paid_media']],
        'shapes': [summary_dict[col]['shapes'] for col in mmm_dashboard_config['paid_media']],
        'scales': [summary_dict[col]['scales'] for col in mmm_dashboard_config['paid_media']],
        'expSpendUnitTotal': exp_spend_unit_total
    }

    paid_media_spend_traspose = mmm_dashboard_config['df'][
        mmm_dashboard_config['paid_media']].transpose().values.tolist()

    target_value = budget_allocator.target_value if budget_allocator.target_value else None

    budget_allocator_spend = budget_allocation(paid_media_spends, budget_allocator.channel_constr_low_sorted,
                                               budget_allocator.channel_constr_up_sorted, periods_to_estimate,
                                               budget_allocator.maxeval, budget_allocator.xtol_rel, eval_list,
                                               paid_media_spend_traspose, rollingWindowLength, target_value,
                                               mmm_dashboard_config['response'])

    return budget_allocator_spend




def append_optimal_spend(mmm_dashboard_config, optimal_spend, periods, date_max):
    """
    Append optimal spend to a DataFrame for a given set of channels and periods.

    Parameters:
    df (pandas.DataFrame): The original DataFrame.
    channels (list): The list of channel names.
    optimal_spend (float): The optimal spend value.
    periods (int): The number of periods to append.

    Returns:
    pandas.DataFrame: The extended DataFrame with optimal spend appended.
    """
    df_optim = mmm_dashboard_config['df'].copy()
    df_tail = df_optim.loc[df_optim[mmm_dashboard_config['date_var']] <= date_max]
    df_extended = df_tail[mmm_dashboard_config['paid_media']].copy()
    new_rows = pd.DataFrame([optimal_spend] * periods, columns=mmm_dashboard_config['paid_media'])
    df_extended = pd.concat([df_extended, new_rows], ignore_index=True)
    return df_extended


def adjust_total(data, type_aggregation):
    """
    Adjusts expected spend days and calculates budget start and end dates for 'total' form type based on the type of aggregation.

    Args:
        data (dict): The form data containing budget allocation parameters.
        type_aggregation (str): The type of aggregation for the budget allocation.

    Returns:
        tuple: A tuple containing the adjusted form data, expected spend days by type of aggregation, and periods to estimate.
    """

    budget_start = datetime.strptime(
        str(data['budget_start']), '%Y-%m-%d')
    budget_end = datetime.strptime(
        str(data['budget_end']), '%Y-%m-%d')

    if type_aggregation == 'Daily':
        expected_spend_days_by_type_aggregation = (
                budget_end - budget_start + timedelta(days=1)).days

        expected_spend_days = periods_to_estimate = expected_spend_days_by_type_aggregation

    elif type_aggregation == 'Weekly':

        dates_weekly = adjust_dates_for_weekly_aggregation(
            budget_start, budget_end)
        budget_start = dates_weekly['date_min']
        budget_end = dates_weekly['date_max']

        expected_spend_days = (
                budget_end - budget_start + timedelta(days=1)).days
        periods_to_estimate = calculate_allocation_expected_spend_days(
            expected_spend_days, {'type_aggregation': 'Weekly'})
        expected_spend_days_by_type_aggregation = periods_to_estimate

    elif type_aggregation == 'Monthly':
        dates_monthly = adjust_dates_for_monthly_aggregation(
            budget_start, budget_end)
        budget_start = dates_monthly['date_min']
        budget_end = dates_monthly['date_max']

        expected_spend_days = (
                budget_end - budget_start + timedelta(days=1)).days
        periods_to_estimate = calculate_allocation_expected_spend_days(
            expected_spend_days, {'type_aggregation': 'Monthly'})
        expected_spend_days_by_type_aggregation = periods_to_estimate

    data_expected_spend = int(data['expected_spend'])

    return expected_spend_days, expected_spend_days_by_type_aggregation, periods_to_estimate, data_expected_spend


def budget_allocation(spends, lb, ub, expected_spend_days, maxeval, xtol_rel, eval_list, df_list, rollingWindowLength,
                      target_value=None, target_var_type=None):
    """
    Perform budget allocation optimization using the COBYLA algorithm.

    Args:
        spends (list): List of initial budget spends.
        lb (list): List of lower bounds for budget spends.
        ub (list): List of upper bounds for budget spends.
        expected_spend_days (int): Expected number of spend days.
        maxeval (int): Maximum number of function evaluations.
        xtol_rel (float): Relative tolerance for convergence.
        eval_list (dict): Dictionary containing evaluation data.
        df_list (list): List of dataframes for evaluation.
        target_value (float, optional): Target value for optimization. Defaults to None.
        target_var_type (str, optional): Type of target variable. Defaults to None.

    Returns:
        list: Optimized budget spends.
    """
    warnings.filterwarnings("ignore")

    expSpendUnitTotal = eval_list["expSpendUnitTotal"]

    def eval_f(_spends, grad):
        """
        Evaluate the objective function for optimization.

        Parameters:
        _spends (list): A list of spend values.
        grad (numpy.ndarray): The gradient array.

        Returns:
        float: The objective function value.

        """
        X = _spends.copy()
        coefsFiltered = eval_list["coefsFiltered"]
        alphas = eval_list["alphas"]
        shapes = eval_list["shapes"]
        scales = eval_list["scales"]
        gammas = eval_list["gammas"]

        if grad.size > 0:
            # Calculate the gradient only if it's needed (i.e., if the 'grad' array is not empty)
            grad = [fx_gradient(x, list_x, coeff, alpha, gamma, shape, scale, expected_spend_days, rollingWindowLength)
                    for x, list_x, coeff, alpha, gamma, shape, scale in
                    zip(X, df_list, coefsFiltered, alphas, gammas, shapes, scales)]

        if target_value is None:
            total_spend = sum(X)
            total_response = -sum(([
                fx_objective(x, list_x, coeff, alpha, gamma, shape, scale, expected_spend_days, rollingWindowLength) for
                x, list_x, coeff, alpha, gamma, shape, scale in
                zip(X, df_list, coefsFiltered, alphas, gammas, shapes, scales)]))

            return total_response
        else:
            total_response = sum(([
                fx_objective(x, list_x, coeff, alpha, gamma, shape, scale, expected_spend_days, rollingWindowLength) for
                x, list_x, coeff, alpha, gamma, shape, scale in
                zip(X, df_list, coefsFiltered, alphas, gammas, shapes, scales)]))
            total_spend = sum(X)

            if target_var_type != 'Revenue':
                objective_value = round(total_spend / total_response, 2)
            else:
                objective_value = round(total_response / total_spend, 2)

            return abs(float(target_value) - float(objective_value))

    def eval_g_eq(_spends, grad, _expSpendUnitTotal):
        X = _spends.copy()
        total_budget_unit = _expSpendUnitTotal
        constr = sum(X) - total_budget_unit
        grad = np.ones(len(_spends))

        return constr

    def eval_g_eq_effi(_spends, grad, _target_value, _target_var_type):
        """
        Calculate the constraint value for the optimization problem.

        Args:
            _spends (list): List of spend values.
            grad (float): Gradient value.
            _target_value (float): Target value.
            _target_var_type (str): Type of target variable.

        Returns:
            float: Constraint value.

        Raises:
            None
        """
        X = _spends.copy()
        Y = float(_target_value)
        coefsFiltered = eval_list["coefsFiltered"]
        alphas = eval_list["alphas"]
        shapes = eval_list["shapes"]
        scales = eval_list["scales"]
        gammas = eval_list["gammas"]

        total_response = sum(([
            fx_objective(x, list_x, coeff, alpha, gamma, shape, scale, expected_spend_days, rollingWindowLength) for
            x, list_x, coeff, alpha, gamma, shape, scale in
            zip(X, df_list, coefsFiltered, alphas, gammas, shapes, scales)]))
        total_spend = sum(X)

        if _target_var_type != 'Revenue':
            constr = total_spend - total_response * Y
        else:
            constr = total_spend - total_response / Y

        return abs(constr)

    opt = nlopt.opt(nlopt.LN_COBYLA, len(spends))

    if target_value is None:
        lambda_constraint = lambda x, grad: eval_g_eq(x, grad, expSpendUnitTotal)
        opt.add_inequality_constraint(lambda_constraint, xtol_rel)
        opt.add_equality_constraint(lambda_constraint, xtol_rel)
    else:
        lambda_constraint = lambda x, grad: eval_g_eq_effi(x, grad, target_value, target_var_type)
        # opt.add_inequality_constraint(lambda_constraint)
        opt.add_equality_constraint(lambda_constraint, xtol_rel)

    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    opt.set_xtol_rel(xtol_rel)
    opt.set_maxeval(maxeval)

    opt.set_min_objective(eval_f)

    optmSpendUnit = opt.optimize(spends)

    return optmSpendUnit


def fx_gradient(x, list_x, coeff, alpha, gamma, shape, scale, expected_spend_days, rollingWindowLength):
    """
    Calculates the gradient of the function fx.

    Parameters:
    - x: The input value.
    - list_x: A list of historical values.
    - coeff: A coefficient value.
    - alpha: An alpha value.
    - gamma: A gamma value.
    - shape: A shape value.
    - scale: A scale value.
    - expected_spend_days: The number of expected spend days.

    Returns:
    - xOut: The gradient of the function fx.
    """
    x_spends = list_x.copy()
    if len(x_spends) < 900:
        x_spends = x_spends + ([x if not pd.isna(x) else 0] * expected_spend_days)
    else:
        x_spends = x_spends[-28:] + ([x if not pd.isna(x) else 0] * expected_spend_days)
    xAdstocked = adstock(x_spends, shape, scale)['x_decayed']
    x_to_saturation = xAdstocked[-rollingWindowLength:]

    xOut = -coeff * ((alpha * (gamma**alpha) * (x_to_saturation**(alpha - 1))) / (x_to_saturation**alpha + gamma**alpha)**2)
    return xOut


def fx_objective(x, list_x, coeff, alpha, gamma, shape, scale, expected_spend_days, rollingWindowLength):
    """
    Calculate the objective value for a given input.

    Parameters:
    x (float): The input value.
    list_x (list): The list of input values.
    coeff (float): The coefficient value.
    alpha (float): The alpha value.
    gamma (float): The gamma value.
    shape (float): The shape value.
    scale (float): The scale value.
    expected_spend_days (int): The number of expected spend days.

    Returns:
    float: The objective value.
    """
    x_spends = list_x.copy()
    if len(x_spends) < 900:
        x_spends = x_spends + ([x if not pd.isna(x) else 0] * expected_spend_days)
    else:
        x_spends = x_spends[-28:] + ([x if not pd.isna(x) else 0] * expected_spend_days)

    xAdstocked = adstock(x_spends, shape, scale)['x_decayed']
    x_to_saturation = xAdstocked[-rollingWindowLength:]

    xDecomp = [coeff * sh if not pd.isna(sh) else 0 for sh in saturation_hill(x_to_saturation, alpha, gamma)]
    xOut = np.mean(xDecomp[-expected_spend_days:])

    return xOut