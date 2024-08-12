import matplotlib.pyplot as plt
import os

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

paid_media = json_model['InputCollect']['paid_media_spends']
all_vars = json_model['InputCollect']['all_ind_vars']



out_dates = list(df_alldecomp_matrix['date_week'])

dfc = df.copy()
dfc = dfc[dfc['date_week'].isin(out_dates)]
dfc.reset_index(inplace=True, drop=True)

df_adstock = df_adstock[df_adstock['date_week'].isin(out_dates)]
df_adstock.reset_index(inplace=True, drop=True)


# Ensure the output directory exists
output_dir = 'data_analysis_tests'
os.makedirs(output_dir, exist_ok=True)

# List of columns to plot
columns = df_adstock.columns

# Iterate over each column and create two plots

for column in columns:
    # First plot: Original spend and Adstocked spend
    plt.figure(figsize=(12, 6))

    # Plot the original spend from df
    plt.plot(dfc.index, dfc[column], label="Original Spend")

    # Plot the adstocked spend from df_adstock
    plt.plot(df_adstock.index, df_adstock[column], label="Adstocked Spend")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Spend")
    plt.title(f"Original and Adstocked Spend for {column}")
    plt.grid(True)

    # Save the plot as an image
    output_path = os.path.join(output_dir, f"{column}_original_adstock.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory

    # Second plot: Saturation spend
    plt.figure(figsize=(12, 6))

    # Plot the saturation spend from df_saturation
    plt.plot(df_saturation.index, df_saturation[column], label="Saturation Spend")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Spend")
    plt.title(f"Saturation Spend for {column}")
    plt.grid(True)

    # Save the plot as an image
    output_path = os.path.join(output_dir, f"{column}_saturation.png")
    #plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory


from scipy.stats import weibull_min


from cassandra.cas_functions import normalize, extract_coefficients_and_confidence_intervals, extract_hyperparameters, \
    create_summary_dictionary


def plot_weibull(shape, scale, name, output_dir='data_analysis_tests', adstock_path='adstock'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    adstock_dir = os.path.join(output_dir, adstock_path)
    os.makedirs(adstock_dir, exist_ok=True)

    # Generate x values
    x = np.arange(1, 101)  # Let's consider a range of 100 for illustration

    # Calculate Weibull PDF
    scale_trans = np.round(np.quantile(x, scale), 0)
    weibull_pdf = weibull_min.pdf(x, c=shape, scale=scale_trans)
    weibull_pdf_normalized = normalize(weibull_pdf)
    # Plot the Weibull PDF
    plt.figure(figsize=(12, 6))
    plt.plot(x, weibull_pdf_normalized, label=f"Weibull PDF (shape={shape}, scale={scale})")
    plt.xlabel("Time")
    plt.ylabel("Normalized Probability Density")
    plt.title(f"Weibull Distribution for {name} (shape={shape}, scale={scale})")
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(adstock_dir, f"{name}weibull_shape_{shape}_scale_{scale}.png")
    plt.savefig(output_path)
    plt.close()  # Close the plot to free up memory

    print(f"Weibull plot saved at {output_path}")


import numpy as np
import matplotlib.pyplot as plt
import os


def saturation_hill(x, alpha, gamma, x_marginal=None):
    inflexion = (np.min(x) * (1 - gamma)) + (np.max(x) * gamma)  # linear interpolation by dot product
    if x_marginal is None:
        x_scurve = x ** alpha / (x ** alpha + inflexion ** alpha)  # plot(x_scurve) summary(x_scurve)
    else:
        x_scurve = x_marginal ** alpha / (x_marginal ** alpha + inflexion ** alpha)
    return x_scurve


def plot_saturation_hill(alpha, gamma, name, output_dir='data_analysis_tests'):
    # Ensure the output directories exist
    os.makedirs(output_dir, exist_ok=True)
    saturation_dir = 'data_analysis_tests/saturation'
    os.makedirs(saturation_dir, exist_ok=True)

    # Generate x values
    x = np.linspace(0, 100, 1000)  # Let's consider a range from 0 to 100 for illustration

    # Calculate Saturation Hill
    y = saturation_hill(x, alpha, gamma)

    # Plot the Saturation Hill
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label=f"Saturation Hill (alpha={alpha}, gamma={gamma})")
    plt.xlabel("Spend")
    plt.ylabel("Response")
    plt.title(f"Saturation Hill Function for {name} (alpha={alpha}, gamma={gamma})")
    plt.legend()
    plt.grid(True)

    # Save the plot
    #output_path = os.path.join(output_dir, f"{name}_saturation_hill_alpha_{alpha}_gamma_{gamma}.png")
    saturation_path = os.path.join(saturation_dir, f"{name}_saturation_hill_alpha_{alpha}_gamma_{gamma}.png")
    plt.savefig(saturation_path)
    plt.close()  # Close the plot to free up memory

    print(f"Saturation Hill plot saved at {saturation_path}")

coef_dict = extract_coefficients_and_confidence_intervals(json_model['ExportedModel']['summary'], paid_media)
hyper_dict, lambda_value = extract_hyperparameters(json_model['ExportedModel']['hyper_values'])
summary_dict = create_summary_dictionary(all_vars, hyper_dict, coef_dict)

for col in df_adstock.columns:
    if col == 'date_week':
        continue
    shape, scale = summary_dict[col]['shapes'], summary_dict[col]['scales']
    alphas, gammas = summary_dict[col]['alphas'], summary_dict[col]['gammas']

    plot_weibull(shape, scale, col)
    plot_saturation_hill(alphas, gammas, col)

