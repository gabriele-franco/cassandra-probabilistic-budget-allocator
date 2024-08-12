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


def plot_cost_revenue_per_media(df, decomp, media, date_var, type):
    """
    analyze cost vs revenue for a given media

    two ways of analyzing the data:
    if type is 'cost', then the x-axis is cost and the y-axis is revenue
    if type is 'saturation', then the x-axis is cost+adstock+saturation and the y-axis is revenue

    :param df:
    :param decomp:
    :param media:
    :param date_var:
    :param type:
    :return:
    """

    assert type in ['Cost', 'Saturation']

    # do a scatter plot of cost vs revenue
    media_cost = df[[date_var, media]].copy()
    media_revenue = decomp[[date_var, media]].copy()

    # set date as index
    media_cost.set_index(date_var, inplace=True)
    media_revenue.set_index(date_var, inplace=True)

    # do a scatter plot
    import matplotlib.pyplot as plt

    # close previous plt
    plt.close()
    plt.scatter(media_cost[media], media_revenue[media])
    plt.xlabel(type)
    plt.ylabel('Revenue')
    plt.title(f'{type} vs Revenue for ' + media)
    # save in data_viz folder
    plt.savefig(f'data_viz/{type}_revenue_' + media + '.png')

    a=1


def analyze_saturation_revenue_per_media_v0(df_saturation, decomp, media, date_var):
    """
    analyze saturation vs revenue for a given media

    :param df_saturation:
    :param decomp:
    :param media:
    :param date_var:
    :return:
    """

    # do a scatter plot of saturation vs revenue
    media_saturation = df_saturation[[date_var, media]].copy()
    media_revenue = decomp[[date_var, media]].copy()

    # set date as index
    media_saturation.set_index(date_var, inplace=True)
    media_revenue.set_index(date_var, inplace=True)

    # do a scatter plot
    # combine the two dataframes
    media_saturation['revenue'] = media_revenue[media]
    media_saturation.columns = ['saturation', 'revenue']
    # sort by saturation ascending
    media_saturation = media_saturation.sort_values(by='saturation')




    # Step 1: Calculate the difference between each saturation point and its neighbors
    media_saturation['diff'] = media_saturation['saturation'].diff().fillna(0)
    media_saturation['diff'] = np.where(media_saturation['diff'] == 0, np.nan, media_saturation['diff'])

    # Step 2: Calculate the confidence score (inverse of the difference)
    # Small difference means high confidence, large difference means low confidence
    media_saturation['confidence'] = 1 / media_saturation['diff']
    media_saturation['confidence'] = media_saturation['confidence'].fillna(0)

    # Normalize the confidence score between 0 and 1
    media_saturation['confidence'] = (media_saturation['confidence'] - media_saturation['confidence'].min()) / \
                                     (media_saturation['confidence'].max() - media_saturation['confidence'].min())

    # Step 3: Calculate the upper and lower bounds on revenue
    # Define a margin for the bounds based on confidence (this is a simple linear example)
    margin_factor = 0.1  # You can adjust this factor to increase/decrease the bounds
    media_saturation['lower_bound'] = media_saturation['revenue'] - margin_factor * (
                1 - media_saturation['confidence']) * media_saturation['revenue']
    media_saturation['upper_bound'] = media_saturation['revenue'] + margin_factor * (
                1 - media_saturation['confidence']) * media_saturation['revenue']

    # Step 4: View the results
    media_saturation = media_saturation[['saturation', 'revenue', 'lower_bound', 'upper_bound', 'confidence']]

    import matplotlib.pyplot as plt

    # plot the scatter plot for the actual data
    plt.scatter(media_saturation['saturation'], media_saturation['revenue'], label='Actual Revenue', color='blue')

    # plot the upper and lower bounds as lines
    plt.plot(media_saturation['saturation'], media_saturation['lower_bound'], label='Lower Bound', linestyle='--',
             color='red')
    plt.plot(media_saturation['saturation'], media_saturation['upper_bound'], label='Upper Bound', linestyle='--',
             color='green')

    # labeling the axes and title
    plt.xlabel('Saturation')
    plt.ylabel('Revenue')
    plt.title('Saturation vs Revenue for ' + media)

    # add a legend to the plot
    plt.legend()

    # display the plot
    plt.show()

def analyze_saturation_revenue_per_media(df_saturation, decomp, media, date_var):
    """
    analyze saturation vs revenue for a given media

    :param df_saturation:
    :param decomp:
    :param media:
    :param date_var:
    :return:
    """
    df_saturation = df_saturation.copy()
    decomp = decomp.copy()

    # preprocessing
    media_saturation = df_saturation[[date_var, media]].copy()
    media_revenue = decomp[[date_var, media]].copy()
    media_saturation.set_index(date_var, inplace=True)
    media_revenue.set_index(date_var, inplace=True)
    media_saturation['revenue'] = media_revenue[media]
    media_saturation.columns = ['saturation', 'revenue']
    # sort by saturation ascending
    media_saturation = media_saturation.sort_values(by='saturation')

    # Step 1: Calculate the local density using Gaussian Kernel Density Estimation (KDE)
    kde = gaussian_kde(media_saturation['saturation'], bw_method='scott')  # 'scott' is the default bandwidth method
    media_saturation['density'] = kde(media_saturation['saturation'])

    # Step 2: Calculate confidence based on density (higher density = higher confidence)
    media_saturation['confidence'] = (media_saturation['density'] - media_saturation['density'].min()) / \
                                     (media_saturation['density'].max() - media_saturation['density'].min())

    # Step 3: Calculate the upper and lower bounds on revenue
    margin_factor = 0.2  # You can adjust this factor to increase/decrease the bounds
    media_saturation['lower_bound'] = media_saturation['revenue'] - margin_factor * (
            1 - media_saturation['confidence']) * media_saturation['revenue']
    media_saturation['upper_bound'] = media_saturation['revenue'] + margin_factor * (
            1 - media_saturation['confidence']) * media_saturation['revenue']

    # Step 4: View the results
    media_saturation = media_saturation[['saturation', 'revenue', 'lower_bound', 'upper_bound', 'confidence']]

    import matplotlib.pyplot as plt

    # plot the scatter plot for the actual data
    plt.scatter(media_saturation['saturation'], media_saturation['revenue'], label='Actual Revenue', color='blue')

    # plot the upper and lower bounds as lines
    plt.plot(media_saturation['saturation'], media_saturation['lower_bound'], label='Lower Bound', linestyle='--',
             color='red')
    plt.plot(media_saturation['saturation'], media_saturation['upper_bound'], label='Upper Bound', linestyle='--',
             color='green')

    # labeling the axes and title
    plt.xlabel('Saturation')
    plt.ylabel('Revenue')
    plt.title('Saturation vs Revenue for ' + media)

    # add a legend to the plot
    plt.legend()

    # save in data viz folder
    plt.savefig('data_viz/saturation_revenue_confidence' + media + '.png')

    a=1

analyze_saturation_revenue_per_media(df_saturation, df_alldecomp_matrix, paid_media[5], date_var)

plot_cost_revenue_per_media(df_saturation, df_alldecomp_matrix, paid_media[5],date_var, 'Saturation')

plot_cost_revenue_per_media(df_resample, df_alldecomp_matrix, paid_media[5],date_var, 'Cost')

