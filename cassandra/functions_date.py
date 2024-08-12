import math
from datetime import datetime, timedelta

def calculate_allocation_expected_spend_days(expected_spend_days: int, mmm_dashboard_config: dict) -> int:
    """
    Calculates the expected number of spend days for budget allocation based on the aggregation type
    specified in the MMM dashboard configuration. This function adjusts the number of expected spend days
    to fit the aggregation cycle (Daily, Weekly, Monthly) by converting the total expected spend days into
    the corresponding number of days, weeks, or months as appropriate.

    Args:
        expected_spend_days (int): The total number of expected spend days originally planned for the budget allocation.
        mmm_dashboard_config (dict): A dictionary containing configuration details for the MMM dashboard,
                                     particularly the type of data aggregation (Daily, Weekly, Monthly).

    Returns:
        int: The adjusted number of expected spend days based on the aggregation type. For daily aggregation,
             this is the direct number of expected spend days. For weekly aggregation, it is the total expected
             spend days divided by 7 (rounded up to the nearest whole number). For monthly aggregation, it is
             the total expected spend days divided by 30 (rounded up to the nearest whole number), assuming an
             average month length of 30 days.

    Note:
        The function uses mathematical rounding up (ceiling) to ensure that the allocation covers the entire
        expected spend period, even if the total days do not evenly divide into weeks or months.
    """

    if mmm_dashboard_config['type_aggregation'] == 'Daily':
        # No adjustment needed for daily aggregation
        pass

    elif mmm_dashboard_config['type_aggregation'] == 'Weekly':
        expected_spend_days = math.ceil(expected_spend_days / 7)

    elif mmm_dashboard_config['type_aggregation'] == 'Monthly':
        expected_spend_days = math.ceil(expected_spend_days / 30)

    return expected_spend_days

def calculate_start_date_based_on_aggregation(date_max, expected_spend_days, aggregation_type):
    """
    Calculates the start date based on the aggregation type.

    Args:
        date_max (datetime): The maximum date.
        expected_spend_days (int): The number of expected spend days.
        aggregation_type (str): The type of aggregation. Possible values are 'Daily', 'Weekly', or 'Monthly'.

    Returns:
        datetime: The calculated start date.

    Raises:
        ValueError: If the aggregation_type is not one of the allowed values.

    """
    if aggregation_type == 'Daily':
        return date_max - timedelta(days=expected_spend_days - 1)
    elif aggregation_type == 'Weekly':
        return date_max - timedelta(weeks=expected_spend_days - 1)
    elif aggregation_type == 'Monthly':
        return date_max - timedelta(days=(expected_spend_days * 30) - 30)
    else:
        raise ValueError("Invalid aggregation_type. Allowed values are 'Daily', 'Weekly', or 'Monthly'.")

def adjust_date(date, aggregation_type):
    """
    Adjusts the given date based on the aggregation type.

    Parameters:
    date (datetime): The date to be adjusted.
    aggregation_type (str): The type of aggregation.

    Returns:
    datetime: The adjusted date.
    """
    if aggregation_type == 'Daily':
        return date
    elif aggregation_type == 'Weekly':
        return date - timedelta(days=date.weekday())
    elif aggregation_type == 'Monthly':
        return date.replace(day=1)

def adjust_allocation_dates_for_aggregation(date_min, date_max, expected_spend_days, aggregation_type):
    """
    Adjusts the allocation dates based on the aggregation type.

    Args:
        date_min (datetime): The minimum date.
        date_max (datetime): The maximum date.
        expected_spend_days (int): The number of expected spend days.
        aggregation_type (str): The type of aggregation.

    Returns:
        tuple: A tuple containing the adjusted allocation dates.

    Raises:
        ValueError: If the aggregation type is not supported.

    """
    if aggregation_type == 'Daily':
        return {
            'date_min': date_min,
            'date_max': date_max, 
            'date_allocation_min': date_max + timedelta(days=1),
            'date_allocation_max': date_max + timedelta(days=expected_spend_days)
        }
        
    elif aggregation_type == 'Weekly':
        return adjust_dates_for_weekly_aggregation_with_week_to_allocate(date_min, date_max, expected_spend_days)
    elif aggregation_type == 'Monthly':
        return adjust_dates_for_monthly_aggregation_with_months_to_allocate(date_min, date_max, expected_spend_days)
    else:
        raise ValueError("Unsupported aggregation type: {}".format(aggregation_type))

def adjust_dates_for_weekly_aggregation_with_week_to_allocate(date_min: datetime, date_max: datetime, expected_spend_days: int):
    """
    Adjusts the start and end dates for budget allocation to align with weekly aggregation cycles, and calculates 
    the allocation start and end dates based on the expected number of spend days. This function sets the start 
    date (`date_min`) to the previous Monday and the end date (`date_max`) to the next Sunday from the provided 
    dates. It then calculates the allocation start and end dates, assuming each week starts on Monday and ends on Sunday.

    Args:
        date_min (datetime): The initial start date for budget allocation.
        date_max (datetime): The initial end date for budget allocation.
        expected_spend_days (int): The total number of expected spend days, used to calculate the number of weeks for allocation.

    Returns:
        dict: A dictionary containing the adjusted start and end dates, including:
              - 'date_min': Adjusted start date to the previous Monday.
              - 'date_max': Adjusted end date to the next Sunday.
              - 'date_allocation_min': The allocation start date, which is the Monday following `date_max`.
              - 'date_allocation_max': The allocation end date, calculated based on the number of weeks derived from `expected_spend_days`.

    This function is particularly useful for budget allocation processes that operate on a weekly cycle, ensuring that 
    the allocation period starts at the beginning of a week and ends at the end of a week, regardless of the initial 
    date range provided.
    """

    dates_adjusted = adjust_dates_for_weekly_aggregation(date_min, date_max)
    date_min_adjusted = dates_adjusted['date_min']
    date_max_adjusted = dates_adjusted['date_max']

    date_allocation_min, date_allocation_max = calculate_expected_spend_days_dates(date_max_adjusted, expected_spend_days, type_aggregation='Weekly', add_days=True)

    return {
        'date_min': date_min_adjusted,
        'date_max': date_max_adjusted,
        'date_allocation_min': date_allocation_min,
        'date_allocation_max': date_allocation_max
    }

def adjust_dates_for_monthly_aggregation_with_months_to_allocate(date_min: datetime, date_max: datetime, expected_spend_days: int):
    """
    Adjusts the start and end dates for budget allocation to align with monthly cycles and calculates the allocation 
    period based on the expected number of spend days. This function first adjusts the start (date_min) and end (date_max) 
    dates to the beginning and end of their respective months. It then sets the allocation start date to the first day 
    of the month following the adjusted end date and calculates the allocation end date based on the expected spend days, 
    assuming an average month length for the calculation.

    Args:
        date_min (datetime): The original start date for the period of interest.
        date_max (datetime): The original end date for the period of interest.
        expected_spend_days (int): The total number of expected spend days, used to derive the number of months for allocation.

    Returns:
        dict: A dictionary containing the adjusted start and end dates, including:
              - 'date_min': The adjusted start date to the first day of the month of the original start date.
              - 'date_max': The adjusted end date to the last day of the month of the original end date.
              - 'date_allocation_min': The allocation start date, set to the first day of the month following the adjusted end date.
              - 'date_allocation_max': The allocation end date, calculated by adding the derived number of months to the allocation start date.

    Note:
        The number of months to allocate is directly derived from the expected spend days provided. The function assumes an average 
        month length of 30 days for this calculation, ensuring that the allocation covers the entire expected spend period as closely as possible.
    """
    dates_adjusted = adjust_dates_for_monthly_aggregation(date_min, date_max)
    date_min_adjusted = dates_adjusted['date_min']
    date_max_adjusted = dates_adjusted['date_max']

    date_allocation_min, date_allocation_max = calculate_expected_spend_days_dates(date_max_adjusted, expected_spend_days, type_aggregation='Monthly', add_days=True)

    return {
        'date_min': date_min_adjusted,
        'date_max': date_max_adjusted,
        'date_allocation_min': date_allocation_min,
        'date_allocation_max': date_allocation_max
    }

def format_dates(date_min, date_max, date_allocation_min, date_allocation_max):
    """
    Formats dates into required string formats.

    Args:
        date_min (datetime.datetime): The minimum date.
        date_max (datetime.datetime): The maximum date.
        date_allocation_min (datetime.datetime): The minimum allocation date.
        date_allocation_max (datetime.datetime): The maximum allocation date.

    Returns:
        dict: A dictionary containing the formatted dates. The keys are as follows:
            - 'date_min': The minimum date formatted as '%Y-%m-%d'.
            - 'date_max': The maximum date formatted as '%Y-%m-%d'.
            - 'date_allocation_min': The minimum allocation date formatted as '%Y-%m-%d'.
            - 'date_allocation_max': The maximum allocation date formatted as '%Y-%m-%d'.
            - 'date_min_formatted_dby': The minimum date formatted as '%d %b %Y'.
            - 'date_max_formatted_dby': The maximum date formatted as '%d %b %Y'.
            - 'date_allocation_min_formatted_dby': The minimum allocation date formatted as '%d %b %Y'.
            - 'date_allocation_max_formatted_dby': The maximum allocation date formatted as '%d %b %Y'.
    """
    formatted_dates = {}
    for key, date in {'date_min': date_min, 'date_max': date_max, 
                      'date_allocation_min': date_allocation_min, 'date_allocation_max': date_allocation_max}.items():
        formatted_dates[key] = date.strftime('%Y-%m-%d')
        formatted_dates[f"{key}_formatted_dby"] = date.strftime('%d %b %Y')
    return formatted_dates

def adjust_dates_for_weekly_aggregation(date_min: datetime, date_max: datetime):
    """
    Adjusts the given start (date_min) and end (date_max) dates to align with the start and end of a weekly cycle. 
    The start date is adjusted to the previous Monday, and the end date is adjusted to the following Sunday. 
    This adjustment ensures that the date range fully encompasses a week or multiple weeks, starting from Monday 
    and ending on Sunday, regardless of the original dates provided.

    Args:
        date_min (datetime): The original start date for the period of interest.
        date_max (datetime): The original end date for the period of interest.

    Returns:
        dict: A dictionary containing the adjusted start and end dates:
              - 'date_min': The adjusted start date, set to the previous Monday of the original start date.
              - 'date_max': The adjusted end date, set to the next Sunday following the original end date.

    This function is particularly useful for scenarios where budget allocations, event planning, or any other 
    operations need to be aligned with standard weekly cycles, ensuring consistency in weekly reporting, 
    analysis, or scheduling tasks.
    """

    # Adjust date_min to the previous Monday
    days_to_previous_monday = date_min.weekday()  # Monday is 0
    date_min_adjusted = date_min - timedelta(days=days_to_previous_monday)

    # Adjust date_max to the next Sunday
    days_to_next_sunday = 6 - date_max.weekday()  # Sunday is 6
    date_max_adjusted = date_max + timedelta(days=days_to_next_sunday)

    return {
        'date_min': date_min_adjusted,
        'date_max': date_max_adjusted,
    }

def adjust_dates_for_monthly_aggregation(date_min: datetime, date_max: datetime):
    """
    Adjusts the provided start (date_min) and end (date_max) dates to align with the beginning and end of their respective months.
    The start date is adjusted to the first day of its month, and the end date is adjusted to the last day of its month. This adjustment
    ensures that the date range covers entire months, starting from the first day of the start month and ending on the last day of the end month.

    Args:
        date_min (datetime): The original start date for the period of interest.
        date_max (datetime): The original end date for the period of interest.

    Returns:
        dict: A dictionary containing the adjusted start and end dates:
              - 'date_min': The adjusted start date, set to the first day of the month of the original start date.
              - 'date_max': The adjusted end date, set to the last day of the month of the original end date.

    This function is particularly useful for scenarios where budget allocations, event planning, or any other operations need to be
    aligned with monthly cycles, ensuring consistency in monthly reporting, analysis, or scheduling tasks. By adjusting dates to cover
    full months, it simplifies period-over-period comparisons and consolidations.
    """
    # Adjust date_min to the first day of its month
    date_min_adjusted = date_min.replace(day=1)

    # Adjust date_max to the last day of its month
    date_max_adjusted = date_max.replace(
        day=1) + relativedelta(months=1) - timedelta(days=1)

    return {
        'date_min': date_min_adjusted,
        'date_max': date_max_adjusted
    }