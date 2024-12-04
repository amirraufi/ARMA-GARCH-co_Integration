# To smoothly run the code please be sure to have the following libraries installed
# Pandas, Numpy, matplotlib, scipy, statsmodels, ARCH, Itertools, Warning, PLOTLY, Seaborn, and NBFORMAT=>4.2 for the last graph



import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.stattools import acf,pacf, adfuller,q_stat           #autocorrelation function and Q-statistics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf             # Functions to plot autocorrelation and partial autocorrelation
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch         # Ljung-Box test for checking autocorrelation in time series residuals
from statsmodels.tsa.arima.model import ARIMA                             # ARIMA model for time series forecasting (AutoRegressive Integrated Moving Average)
from scipy.stats import jarque_bera
from scipy.stats import norm                                              # Norm for the normal distribution in statistical analysis
from arch import arch_model
import itertools
import warnings
from scipy.stats import skew
from statsmodels.tools.sm_exceptions import  ValueWarning                 # Handling specific warning types from statsmodels
from arch.univariate import ZeroMean, ConstantMean
import plotly.graph_objects as go
import math
import pandas as pd  # For handling and manipulating datasets using DataFrames
import numpy as np  # For numerical computations and array operations
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For advanced data visualizations built on top of matplotlib
import scipy.stats as scs  # For statistical functions and tests like distributions and significance testing
import warnings  # For handling warnings in the code
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # Johansen cointegration test
from statsmodels.tsa.vector_ar.vecm import VECM  # Vector Error Correction Model
from statsmodels.tsa.vector_ar.vecm import select_order  # Order selection for VECM
from statsmodels.tsa.vector_ar.var_model import VAR  # Vector Autoregression Model
from scipy.stats import jarque_bera

# Loading the Data, Calculating Returns, and Ploting
# As we had to send one file for the code and not a folder I did not add the data and making a folder so please change the data link for a smooth run

data_1_link = "/Users/amirhossein/Downloads/StockIndexFXDATA.csv"
data = pd.read_csv(data_1_link)
#print(data.head())
data["DATE"]= pd.to_datetime(data["DATE"], format= "%d/%m/%Y")
data.set_index("DATE", inplace=True)
# Return Function 
def compute_return(data, columns):
    # generalising for more than one variable 
    for column in columns:
        
        # Calculate log-returns
        data[f'log_returns_{column}'] = np.log((data[column] / data[column].shift(1)))*100
        
        # Calculate absolute log-returns
        data[f'abs_log_returns_{column}'] = np.abs(data[f'log_returns_{column}'])
        
        # Calculate squared log-returns
        data[f'squared_log_returns_{column}'] = data[f'log_returns_{column}'] ** 2
        
        # Calculate standardised log-returns
        std_dev = data[f'log_returns_{column}'].std()
        data[f'std_log_returns_{column}'] = data[f'log_returns_{column}'] / std_dev
        
    df_returns = data.dropna()

    return df_returns

data_1 = compute_return(data, ["VIX", "SP500"])
#print(data_1.head)

Instruments_used = ["VIX", "SP500"]

# Create a figure and axes grid with 5 rows and 2 columns, with a figure size of 18,24 inches
fig, ax = plt.subplots(5,2, figsize=(18, 24))
columns = Instruments_used
for i in range(len(columns)):
    col = columns[i]  # Set the current column as a variable 

    # Plot original price series
    ax[0, i].plot(data_1[col], color='blue')
    ax[0, i].set_title(f'{col}')
    ax[0, i].grid(True, linestyle=":")

    # Plot log-returns
    ax[1, i].plot(data_1[f'log_returns_{col}'], color='red')
    ax[1, i].set_title(f'Log Returns {col}')
    ax[1, i].grid(True, linestyle=":")

    # Plot standardized log-returns
    ax[2, i].plot(data_1[f'std_log_returns_{col}'], color='purple')
    ax[2, i].set_title(f'Standardized Log Returns {col}')
    ax[2, i].grid(True, linestyle=":")


    # Plot squared log-returns
    ax[3, i].plot(data_1[f'squared_log_returns_{col}'], color='orange')
    ax[3, i].set_title(f'Squared Log Returns {col}')
    ax[3, i].grid(True, linestyle=":")

    # Plot absolute log-returns
    ax[4, i].plot(data_1[f'abs_log_returns_{col}'], color='purple')
    ax[4, i].set_title(f'Absolute Log Returns {col}')
    ax[4, i].grid(True, linestyle=":")

plt.tight_layout()  
plt.show()
# Saving the graph
#plt.savefig('returns_plot.png') 

#-------------------------------------------------------------------------------------------------------------------------------------#
#descibing the data related to Q1 part a


data_2 =data_1[['SP500', 'VIX', 
                          'log_returns_SP500', 'std_log_returns_SP500', 'abs_log_returns_SP500', 'squared_log_returns_SP500',
                          'log_returns_VIX', 'std_log_returns_VIX', 'abs_log_returns_VIX', 'squared_log_returns_VIX' ]]
descriptive_stats = data_2.describe()
# Calculate skewness and kurtosis for the selected columns 
skewness = data_2.skew()
kurtosis = data_2.kurtosis() + 3
descriptive_stats.loc['skew'] = skewness
descriptive_stats.loc['kurtosis'] = kurtosis

num_lags = 1 # Specify the number of lags for ACF calculation
for column in data_2:
    acf_values = acf(data_2[column].dropna(), nlags=num_lags, fft=False)  
    #Append the ACF values for each lag to the descriptive statistics DataFrame
    for lag in range(1, num_lags + 1):
        descriptive_stats.loc[f'acf_lag_{lag}', column] = acf_values[lag]
print(descriptive_stats.round(4))
desc_stat = pd.DataFrame(descriptive_stats.round(3))
# Saving the data 
#desc_stat.to_csv('/Users/amirhossein/Downloads/desc_stat.csv', index=False)

#-------------------------------------------------------------------------------------------------------------------------------------#
# Testing for Normal Distribution using jarque_bera related to Q1 part a

col_1 = 'log_returns_VIX'
col_2 = 'log_returns_SP500'
# Remove any missing values from the 'log_returns_FTSE100' series
series_1 = data_2[col_1].dropna()
series_2 = data_2[col_2].dropna()
# Plot histogram
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Plot a histogram of the series with 30 bins, blue color, and transparency level of 0.75
ax.hist(series_1, bins=30, alpha=0.75, color='blue')
ax.set_title(f'{col_1} Histogram')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.grid(True, linestyle=":")
plt.show()

# Jarque-Bera test for normality
jb_stat, p_value = jarque_bera(series_1)

# Determine if the series is normally distributed. If p-value >= 0.05, we fail to reject the null hypothesis of normality
is_normal = "Yes" if p_value >= 0.05 else "No"

# Print the results of the Jarque-Bera test: the test statistic, p-value, and whether the series is normally distributed
print('Jarque-Bera normality test for %s: JB stat = %.3f, p-value = %.3f' %(col_1,jb_stat,p_value))
print('Is %s normally distributed? %s' % (col_1,is_normal))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))

# Plot a histogram of the series with 30 bins, blue color, and transparency level of 0.75
ax.hist(series_2, bins=30, alpha=0.75, color='blue')
ax.set_title(f'{col_2} Histogram')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.grid(True, linestyle=":")
plt.show()

# Jarque-Bera test for normality
jb_stat, p_value = jarque_bera(series_2)

# Determine if the series is normally distributed. If p-value >= 0.05, we fail to reject the null hypothesis of normality
is_normal = "Yes" if p_value >= 0.05 else "No"

# Print the results of the Jarque-Bera test: the test statistic, p-value, and whether the series is normally distributed
print('Jarque-Bera normality test for %s: JB stat = %.3f, p-value = %.3f' %(col_2,jb_stat,p_value))
print('Is %s normally distributed? %s' % (col_2,is_normal))


#-------------------------------------------------------------------------------------------------------------------------------------#
#Autocorrologram related to Q1 part a and b

# Aucorrologram and partial Autocorrologram 
data_3 = data_2[['SP500', 'VIX','log_returns_SP500','squared_log_returns_SP500','log_returns_VIX','squared_log_returns_VIX']]

columns = data_3.columns[len(Instruments_used):] 

# Creating a grid of subplots with a number of rows equal to the number of columns in 'columns' and two columns (ACF and PACF) 
# width is fixed and the hight is dynamically adjusted by the number of rows
fig, ax = plt.subplots(nrows=len(columns), ncols=2, figsize=(10, 3 * len(columns)))  

# Looping through each column to plot ACF and PACF side by side for each variable
for i in range(len(columns)):
    col = columns[i]  # Get the column name (either log return or squared log return)

    # Plotting the ACF for the current column 
    plot_acf(data_3[col], lags=30, ax=ax[i, 0])  # Plot ACF for up to 30 lags
    ax[i, 0].set_title(f'ACF of {col}')  # Setting the title for the ACF plot
    ax[i, 0].grid(True, linestyle=":")
    # Plotting the PACF for the current column
    plot_pacf(data_3[col], lags=30, ax=ax[i, 1])  # Plot PACF for up to 30 lags
    ax[i, 1].set_title(f'PACF of {col}')  # Setting the title for the PACF plot
    ax[i, 1].grid(True, linestyle=":")
# Adjust the layout so that the subplots do not overlap
plt.tight_layout()  # Automatically adjusts subplot parameters to give some padding between plots
plt.show()  # Display the plots



#-------------------------------------------------------------------------------------------------------------------------------------#
#Ljung-Box Test to see if there is any serial correlation in the error term up to order k  related to Q1 part a and b

results_df = acorr_ljungbox(series_1, lags=10, return_df=True)

print(round(results_df,5))

results_df_2 = acorr_ljungbox(series_2, lags=10, return_df=True)

print(round(results_df_2,5))
    

columns_r = ['log_returns_SP500', 'squared_log_returns_SP500', 
            'log_returns_VIX', 'squared_log_returns_VIX']
results = []

# Loop over each column in columns_r
for col in columns_r:
    # Number of lags to consider
    nlags = 10
    
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(index=range(1, nlags + 1))
    
    # Calculate ACF values (excluding lag 0)
    acf_values = acf(data_3[col], nlags=nlags)[1:]
    
    # Calculate PACF values (excluding lag 0)
    pacf_values = pacf(data_3[col], nlags=nlags)[1:]
    
    # Perform the Ljung-Box test
    lb_test = acorr_ljungbox(data_3[col], lags=nlags, return_df=True)
    
    # Store the results in the DataFrame
    results_df[f'{col}_ACF'] = acf_values
    results_df[f'{col}_PACF'] = pacf_values
    results_df[f'{col}_LB_Q_Stat'] = lb_test['lb_stat']
    results_df[f'{col}_LB_p_Value'] = lb_test['lb_pvalue']
    
    # Round the results to 4 decimal places
    results_df = results_df.round(4)
    
    # Append the results DataFrame to the results list
    results.append(results_df)


combined_results = pd.concat(results, axis=1)
#print(combined_results.head())
combined_results.to_csv("/Users/amirhossein/Downloads/comb_results.csv")

#data_3.to_csv("/Users/amirhossein/Downloads/data3.csv")

### There is no serial Correlation based on the results of the test up to order of 10 ###

#-------------------------------------------------------------------------------------------------------------------------------------#
#Checking for staionarity related to Q1 part a and b
in_sample_end_date = '2021-12-31'
data_in_sample = data_3.loc[:in_sample_end_date]
#print(data_in_sample.head())

series1 = data_in_sample['log_returns_SP500']
series2 = data_in_sample['log_returns_VIX']

def adf_test(series, name):
    print(f'Performing ADF Test on {name}')  
    # The ADF test is applied to the series after dropping any missing values
    # 'maxlag=20' sets the maximum number of lags to the test
    result = adfuller(series.dropna(), maxlag=20)
    print('ADF Statistic: %f' % result[0])  # Display the ADF test statistic
    print('p-value: %f' % result[1])  # Display the p-value of the test
    print('Critical Values:')  # Display the critical values for different significance levels
    for key, value in result[4].items():  # Iterate over the dictionary of test statistic values
        print(f'\t{key}: {value}')  # Print each critical value
    print()  # Print a newline for better readability between outputs


# Perform ADF test
adf_test(series1, 'Series 1')
adf_test(series2, 'Series 2')
# series are stationary and do not need transfomation





#-------------------------------------------------------------------------------------------------------------------------------------#
#Finding the Best Arma Model Question 1 Part b





def select_top_arima_models(series, max_p=3, max_q=3, d=0, top_n=5):
   """ Selecting the top 5 moodels based on AIC, BIC, HQIC 
        
   """
    results = []

    # Iterate over all combinations of p and q
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        if p == 0 and q == 0:
            continue  # Skip the (0,0) model
        try:
            model = ARIMA(series, order=(p, d, q))
            model_fit = model.fit()
            aic = model_fit.aic
            bic = model_fit.bic
            hqic = model_fit.hqic
            results.append({
                'AR Order (p)': p,
                'MA Order (q)': q,
                'AIC': aic,
                'BIC': bic,
                'HQIC': hqic,
                'Model Summary': model_fit.summary().tables[0].as_html()  # Optional: Store summary as HTML
            })
        except Exception as e:
            print(f"ARIMA({p},{d},{q}) model fitting failed: {e}")
            continue  # Skip non-converging models

    # Convert results to DataFrame
    all_results = pd.DataFrame(results)

    if all_results.empty:
        print("No ARIMA models were successfully fitted.")
        return None, None, None, all_results

    # Sort and select top_n for each criterion
    top_aic = all_results.nsmallest(top_n, 'AIC').reset_index(drop=True)
    top_bic = all_results.nsmallest(top_n, 'BIC').reset_index(drop=True)
    top_hqic = all_results.nsmallest(top_n, 'HQIC').reset_index(drop=True)

    return top_aic, top_bic, top_hqic, all_results
def display_top_arima_models(series_name, top_aic, top_bic, top_hqic):
    """
    Displays the top ARIMA models based on AIC, BIC, and HQIC.
    Returns:
    - combined_top (pd.DataFrame): Combined DataFrame of top models.
    """
    print(f"\n=== Top {len(top_aic)} ARIMA Models for {series_name} ===")
    
    # Display Top AIC Models
    print("\n--- Top Models Based on AIC ---")
    print(top_aic[['AR Order (p)', 'MA Order (q)', 'AIC', 'BIC', 'HQIC']].to_string(index=False))
    
    # Display Top BIC Models
    print("\n--- Top Models Based on BIC ---")
    print(top_bic[['AR Order (p)', 'MA Order (q)', 'AIC', 'BIC', 'HQIC']].to_string(index=False))
    
    # Display Top HQIC Models
    print("\n--- Top Models Based on HQIC ---")
    print(top_hqic[['AR Order (p)', 'MA Order (q)', 'AIC', 'BIC', 'HQIC']].to_string(index=False))
    
    # Optionally, combine all top models into one DataFrame with a Criterion column
    top_aic_copy = top_aic.copy()
    top_aic_copy['Criterion'] = 'AIC'
    top_bic_copy = top_bic.copy()
    top_bic_copy['Criterion'] = 'BIC'
    top_hqic_copy = top_hqic.copy()
    top_hqic_copy['Criterion'] = 'HQIC'
    
    combined_top = pd.concat([top_aic_copy, top_bic_copy, top_hqic_copy], ignore_index=True)
    
    return combined_top
# Define the cutoff date for in-sample data
in_sample_end_date = '2021-12-31'
data_in_sample = data_3.loc[:in_sample_end_date]

# Select the two series
series1 = data_in_sample['log_returns_SP500']
series2 = data_in_sample['log_returns_VIX']

# Select Top ARIMA Models for Series 1 (SP500)
print("Selecting top ARIMA models for Series 1 (log_returns_SP500)...")
top_aic_sp500, top_bic_sp500, top_hqic_sp500, all_results_sp500 = select_top_arima_models(
    series1, 
    max_p=3, 
    max_q=3, 
    d=0, 
    top_n=5
)

# Display Top ARIMA Models for Series 1
combined_top_sp500 = display_top_arima_models(
    "Series 1 (log_returns_SP500)", 
    top_aic_sp500, 
    top_bic_sp500, 
    top_hqic_sp500
)

# Select Top ARIMA Models for Series 2 (VIX)
print("\nSelecting top ARIMA models for Series 2 (log_returns_VIX)...")
top_aic_vix, top_bic_vix, top_hqic_vix, all_results_vix = select_top_arima_models(
    series2, 
    max_p=3, 
    max_q=3, 
    d=0, 
    top_n=5
)

# Display Top ARIMA Models for Series 2
combined_top_vix = display_top_arima_models(
    "Series 2 (log_returns_VIX)", 
    top_aic_vix, 
    top_bic_vix, 
    top_hqic_vix
)

best_models_combined = pd.concat([combined_top_sp500, combined_top_vix], ignore_index=True)
print(best_models_combined)
# Save the combined top models to CSV
best_models_combined.to_csv('/Users/amirhossein/Downloads/top_arima_models_info.csv', index=False)

print("\nTop ARIMA models have been saved to '/Users/amirhossein/Downloads/top_arima_models_info.csv'.")


#Suppress warnings for cleaner output
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


#================================#
# Finding the Best order for the GARCH Componenet of the model

def estimate_arma_garch(series, arma_order=(3, 3), garch_order=(1, 1), distribution='normal', print_output=True):
   
    results = {}
    garch_info = {}

    # Step 1: Fit the ARMA(3,3) model
    print(f"Fitting ARMA{arma_order} model...")
    try:
        arma_model = ARIMA(series, order=(arma_order[0], 0, arma_order[1]))
        arma_fit = arma_model.fit()
        print("ARMA model fitted successfully.\n")
        if print_output:
            print(arma_fit.summary())
    except Exception as e:
        print(f"ARMA{arma_order} model fitting failed: {e}")
        return None, None  # Return None if ARMA fitting fails

    # Extract residuals
    residuals = arma_fit.resid.dropna()

    # Step 2: Fit the GARCH(p, q) model
    print(f"Fitting GARCH{garch_order} model...")
    try:
        garch = arch_model(
            residuals,
            mean='Zero',
            vol='GARCH',
            p=garch_order[0],
            q=garch_order[1],
            dist=distribution
        )
        garch_fit = garch.fit(disp='off')
        print("GARCH model fitted successfully.\n")
        if print_output:
            print(garch_fit.summary())

        # Compute HQIC manually
        loglike = garch_fit.loglikelihood
        k = len(garch_fit.params)
        n = len(residuals)
        hqic = -2 * loglike + 2 * k * np.log(np.log(n))

        # Record GARCH information
        garch_info = {
            'GARCH Order (p, q)': garch_order,
            'AIC': garch_fit.aic,
            'BIC': garch_fit.bic,
            'HQIC': hqic
        }

        results = {
            'ARMA_Fit': arma_fit,
            'GARCH_Fit': garch_fit
        }

    except Exception as e:
        print(f"GARCH{garch_order} model fitting failed: {e}")
        return results, None  # Return ARMA fit and None if GARCH fitting fails

    return results, garch_info
def iterate_garch_orders_fixed_arma(series, fixed_arma_order=(3, 3), max_p=3, max_q=3, distribution='normal'):
   
    results = []

    print(f"Iterating through GARCH(p, q) orders with fixed ARMA{fixed_arma_order}...\n")

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            print(f"Fitting ARMA{fixed_arma_order}-GARCH({p}, {q}) model...")
            arma_garch_results, garch_info = estimate_arma_garch(
                series=series,
                arma_order=fixed_arma_order,
                garch_order=(p, q),
                distribution=distribution,
                print_output=False  # Set to True to see model summaries
            )
            if garch_info:
                results.append({
                    'GARCH Order (p, q)': f'({p}, {q})',
                    'AIC': garch_info['AIC'],
                    'BIC': garch_info['BIC'],
                    'HQIC': garch_info['HQIC']
                })
                print(f"Completed GARCH({p}, {q}): AIC={garch_info['AIC']:.2f}, BIC={garch_info['BIC']:.2f}, HQIC={garch_info['HQIC']:.2f}\n")
            else:
                print(f"GARCH({p}, {q}) model fitting failed or HQIC not available.\n")
                continue  # Skip if GARCH fitting failed

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("\nNo GARCH models were successfully fitted.")
    else:
        print("\nAll GARCH models have been fitted successfully.")

    return results_df
def find_best_garch_model(results_df, criterion='AIC'):
   
    if results_df.empty:
        print("No results to evaluate.")
        return None

    # Sort the DataFrame based on the specified criterion
    sorted_df = results_df.sort_values(by=criterion).reset_index(drop=True)

    # Select the top model
    best_model = sorted_df.iloc[0].to_dict()

    print(f"\nBest GARCH model based on {criterion}:")
    print(f"Order (p, q): {best_model['GARCH Order (p, q)']}")
    print(f"{criterion}: {best_model[criterion]:.2f}")

    return best_model


# Define the cutoff date for in-sample data
in_sample_end_date = '2021-12-31'
data_in_sample = data_3.loc[:in_sample_end_date]

# Select the two series
series_sp500 = data_in_sample['log_returns_SP500']
series_vix = data_in_sample['log_returns_VIX']

# Iterate and fit GARCH models for SP500
print("=== SP500 Series ===")
garch_results_sp500 = iterate_garch_orders_fixed_arma(
    series=series_sp500,
    fixed_arma_order=(3, 3),
    max_p=3,
    max_q=3,
    distribution='normal'
)

# Display the results
print("\nGARCH Model Selection Results for SP500:")
print(garch_results_sp500)

# Save the results to CSV
garch_results_sp500.to_csv('/Users/amirhossein/Downloads/garch_results_sp500.csv', index=False)

# Find the best GARCH model for SP500 based on AIC
best_garch_sp500 = find_best_garch_model(garch_results_sp500, criterion='AIC')

# Iterate and fit GARCH models for VIX
print("\n=== VIX Series ===")
garch_results_vix = iterate_garch_orders_fixed_arma(
    series=series_vix,
    fixed_arma_order=(3, 3),
    max_p=3,
    max_q=3,
    distribution='normal'
)

# Display the results
print("\nGARCH Model Selection Results for VIX:")
print(garch_results_vix)

# Save the results to CSV
garch_results_vix.to_csv('/Users/amirhossein/Downloads/garch_results_vix.csv', index=False)

# Find the best GARCH model for VIX based on AIC
best_garch_vix = find_best_garch_model(garch_results_vix, criterion='AIC')




""" Best Model found for VIX is ARMA(3,3)Garch(1,1)
    Best Model found for SP500 is ARMA(3,3)GArch(2,2) """





#-------------------------------------------------------------------------------------------------------------------------------------#
#Discussing my model






vix, sp500 = "VIX", "SP500"
def estimate_arma_garch(series, arma_order=(1,1), garch_order=(1,1), distribution='normal', print_output=True):
    
    # Check if there are MA terms in the Mean equation. If not, estimate AR-GARCH in single setp using arch
    if arma_order[1] == 0:
        model = arch_model(series, mean='AR', lags=arma_order[0], vol='GARCH', 
                           p=garch_order[0], q=garch_order[1], dist=distribution)
        garch_fit = model.fit(disp='off')
        if print_output: 
            print(garch_fit.summary())
        
        arma_fit = None # Set arma_fit to None
            
    else: 
        # Perform 2-step estimation
        # Step 1: Estimate ARMA mean equation and extract the residuals
        mean_eq = ARIMA(series, order = (arma_order[0],0,arma_order[1]))
        arma_fit = mean_eq.fit()
        resid = arma_fit.resid.dropna(); resid.name = 'Residuals'
        if print_output: 
            print(arma_fit.summary())

        # Step 2: Fit a Constant-mean GARCH model to the ARMA residuals
        var_eq = arch_model(resid, mean = 'Zero', vol='GARCH', p=garch_order[0], q=garch_order[1], dist=distribution)
        garch_fit = var_eq.fit(disp='off')
        if print_output: 
            print(garch_fit.summary())

    return arma_fit, garch_fit
arma_fit_1, garch_fit_1 = estimate_arma_garch(series1, arma_order=(3,3), garch_order=(2, 2))
arma_fit_2, garch_fit_2 = estimate_arma_garch(series2, arma_order=(3,3), garch_order=(1, 1))


def check_arch_effects_after_garch(garch_fit):
    # Use the residuals of the fitted GARCH model
    std_resid = garch_fit.std_resid
    # std_resid = garch_fit.resid/garch_fit.conditional_volatility # Give same results
    std_resid = std_resid.dropna()
    arch_test = het_arch(std_resid, nlags = 10)
    print(f"Post-GARCH ARCH Test: LM stat = %.3f, p-value = %.3f" %(arch_test[0],arch_test[1]))

# Re-check for ARCH effects
print(f"\n{sp500} Post-GARCH ARCH Test:")
check_arch_effects_after_garch(garch_fit_1)
print(f"\n{vix} Post-GARCH ARCH Test:")
check_arch_effects_after_garch(garch_fit_2)



fig, ax = plt.subplots(1,2, figsize=(14, 4))
    
# Plot fitted conditional standard deviations
ax[0].plot(garch_fit_1.conditional_volatility)
ax[0].set_title(f'{sp500} Conditional Volatility')
ax[0].grid(True,linestyle=":")
    
# Plot fitted conditional variance
ax[1].plot(garch_fit_1.conditional_volatility**2)
ax[1].set_title(f'{sp500} Conditional Variance')
ax[1].grid(True,linestyle=":")
    

def check_standardized_residuals(garch_fit, series_name):
    
    ## Extract standardized residuals
    std_resid = garch_fit.std_resid.dropna()
    std_resid.name = None
    
    # Plot
    fig, axes = plt.subplots(nrows=2,ncols=2,figsize=(16,10))

    ## Time series plot of Standardized residuals
    axes[0 , 0].plot(std_resid)
    axes[0,0].set_title(f'Standardized residuals - {series_name}')
    axes[0,0].grid(True, linestyle=":")

    ## distribution of residuals
    x_lim = (-1.96 * 2, 1.96 * 2)
    x_range = np.linspace(x_lim[0], x_lim[1])
    norm_pdf = stats.norm.pdf(x_range)
    sns.histplot(std_resid,kde=True,stat='density',ax=axes[0,1])
    axes[0,1].plot(x_range, norm_pdf, 'g', lw=2, label='N(0,1)')
    axes[0,1].set_title(f'Distribution of standardized residuals - {series_name}')
    axes[0,1].set_xlim(x_lim)
    axes[0,1].legend()
    axes[0,1].grid(True, linestyle=":")

    # Perform Jarque-Bera test for normality
    jb_stat, jb_p_value = jarque_bera(std_resid)
    print(f'Jarque-Bera Test for {series_name}:')
    print(f'JB Statistic: %.3f; p-value: %.3f' %(jb_stat, jb_p_value))


    ## ACF plot of standardized residual
    plot_acf(std_resid, ax=axes[1,0], alpha=0.05)
    axes[1,0].set_title(f'ACF of standardized residuals - {series_name}')
    axes[1,0].grid(True, linestyle=":")
    

    ## ACF plot of squared standardized residual
    plot_acf(std_resid**2, ax=axes[1,1], alpha=0.05)
    axes[1,1].set_title(f'ACF of standardized residuals squared - {series_name}')
    axes[1,1].grid(True, linestyle=":")
    
print(f"\n{sp500} Standardized Residuals Analysis:")
check_standardized_residuals(garch_fit_1, sp500)


fig, ax = plt.subplots(1,2, figsize=(14, 4))
    
# Plot fitted conditional standard deviations
ax[0].plot(garch_fit_2.conditional_volatility)
ax[0].set_title(f'{vix} Conditional Volatility')
ax[0].grid(True,linestyle=":")
    
# Plot fitted conditional variance
ax[1].plot(garch_fit_2.conditional_volatility**2)
ax[1].set_title(f'{vix} Conditional Variance')
ax[1].grid(True,linestyle=":")

    
print(f"\n{vix} Standardized Residuals Analysis:")
check_standardized_residuals(garch_fit_2, vix)
plt.show()





#-------------------------------------------------------------------------------------------------------------------------------------#
# Alternative Methode Selected for sp500 ARMA(1,1)Garch(2,3) and for VIX ARMA(1,3)GARCH(1,2) now go for forecast q1.c
# Function to estimate ARMA-GARCH and produce forecasts




series_1 = data_3['log_returns_SP500']
series_2 = data_3['log_returns_VIX']
def arma_garch_estimate_and_forecast(ret, cutoff_date, arma_order=(1,1), garch_order=(1, 1), 
                                     distribution='normal', print_output=True):
    # This function estimates ARMA-GARCH model using in-sample/train data and
    # use the estimated model to produce out-of-sample static one-step ahead
    # returns and conditional variance forecasts, without re-estimating ARMA-GARCH model
    # each time a new observation becomes available

    # Create dataframe to store the forecasts
    forecast_df = pd.DataFrame(index=ret.index)
    forecast_df['ret_true'] = ret
    forecast_df['retf'] = ret*np.nan # return forecasts
    forecast_df['varf'] = ret*np.nan # variance forecasts
    
    # Check if there are MA terms in the Mean equation. If not, estimate AR-GARCH in single setp using arch 
    # and produce return and variance forecasts in one go
    if arma_order[1] == 0:
        ## Step 1: Estimate AR-GARCH model
        model = arch_model(ret, mean='AR', lags=arma_order[0], vol='GARCH', 
                           p=garch_order[0], q=garch_order[1], dist=distribution)
        # Estimate the model using data before cutoff_date 
        garch_fit = model.fit(last_obs = cutoff_date, disp='off')
        arma_fit = None
        if print_output: 
            print(garch_fit.summary())
        
        # Produce forecasts
        fcasts = garch_fit.forecast(align='target')
        forecast_df['retf'] = fcasts.mean
        forecast_df['varf'] = fcasts.variance
        
    else:
        # There are MA terms in the Mean Equation. Perform 2-step estimation and forecasts.    

        ## Step 1: Obtain in-sample/train & out-of-sample/test dataset
        ret_train = ret.loc[ret.index < cutoff_date]
        ret_test = ret.loc[ret.index >= cutoff_date]

        ## Step 2: Estimate ARMA-GARCH model using train dataset
        arma_fit, garch_fit = estimate_arma_garch(ret_train, arma_order, garch_order, distribution, print_output)

        ## Step 3: Produce mean forecasts based on ARMA model
        for fdate in ret_test.index:
            # print('%s' % str(fdate))
            # Step a: Get an expanded in-sample data by one obs at a time
            y = ret.loc[ret.index < fdate]

            # Step b: Apply the estimated model to the new data
            arma_fit_ext = arma_fit.apply(y, refit=False)

            # Step c: Produce one-step ahead forecast
            forecast_df.loc[fdate,'retf'] = arma_fit_ext.forecast().iloc[0]


        ## Step 4: Compute return forecasts residuals/errors
        residf = ret_test - forecast_df.loc[ret.index >= cutoff_date]['retf']
        resid = pd.concat([arma_fit.resid.dropna(), residf])

        ## Step 5: Produce variance forecasts
        # Redefine var_eq with return errors over the whole sample period
        var_eq = ZeroMean(y = resid, volatility = garch_fit.model.volatility, distribution = garch_fit.model.distribution)   

        # Re-estimate var_eq using in-sample data
        var_eq_fit = var_eq.fit(last_obs = cutoff_date, disp='off')

        # Produce variance forecasts
        forecast_df['varf'] = var_eq_fit.forecast(align='target').variance

    return forecast_df, arma_fit, garch_fit 

#-------------------------------------------------------------------------------------------------------------------------------------#
#Graphs for the forecast of the best models Q1 part C
cutoff_date = pd.to_datetime('2021-12-31')
forecast_df_1,*_ = arma_garch_estimate_and_forecast(series_1, cutoff_date, arma_order=(3,3), garch_order=(2, 2))
forecast_df_1
forecast_df_2,*_ = arma_garch_estimate_and_forecast(series_2, cutoff_date, arma_order=(3,3), garch_order=(1, 1))
forecast_df_2
fig, ax = plt.subplots(1, 3, figsize = (15,4))
ret_true_1 = forecast_df_1.loc[forecast_df_1.index>=cutoff_date,'ret_true']
retf_1 = forecast_df_1.loc[forecast_df_1.index>=cutoff_date,'retf']
varf_1 = forecast_df_1.loc[forecast_df_1.index>=cutoff_date,'varf']
retf_lb = retf_1 - 1.96*varf_1**0.5 # Lower bound 95%
retf_ub = retf_1 + 1.96*varf_1**0.5 # Upper bound 95%

# ax[0]: Conditional variance forecasts
varf_1.plot(ax=ax[0],style='k',label='Variance forecasts')
ax[0].set_title('One-step conditional Variance forecasts')
ax[0].set_xlim(varf_1.index[0],varf_1.index[-1])
ax[0].legend(loc='best')


# ax[1]: Mean forecast
retf_1.plot(ax=ax[1],style='k',label='Return forecasts')
ax[1].set_title('One-step conditional return forecasts')
ax[1].set_xlim(retf_1.index[0],retf_1.index[-1])
ax[1].legend(loc='best')

# ax[2]: Mean forecast vs true
ret_true_1.plot(ax=ax[2],style='b',label='Observed returns')
retf_1.plot(ax=ax[2],style='k',label='Return forecasts')
retf_lb.plot(ax=ax[2],style='g--',label='95% IC')
retf_ub.plot(ax=ax[2],style='g--',label='')
ax[2].set_title('One-step conditional Return forecasts')
ax[2].set_xlim(retf_1.index[0],retf_1.index[-1])
ax[2].legend(loc='best')

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 3, figsize = (15,4))
ret_true_2 = forecast_df_2.loc[forecast_df_2.index>=cutoff_date,'ret_true']
retf_2 = forecast_df_2.loc[forecast_df_2.index>=cutoff_date,'retf']
varf_2 = forecast_df_2.loc[forecast_df_2.index>=cutoff_date,'varf']
retf_lb = retf_2 - 1.96*varf_2**0.5 # Lower bound 95%
retf_ub = retf_2 + 1.96*varf_2**0.5 # Upper bound 95%

# ax[0]: Conditional variance forecasts
varf_2.plot(ax=ax[0],style='k',label='Variance forecasts')
ax[0].set_title('One-step conditional Variance forecasts')
ax[0].set_xlim(varf_2.index[0],varf_2.index[-1])
ax[0].legend(loc='best')


# ax[1]: Mean forecast
retf_2.plot(ax=ax[1],style='k',label='Return forecasts')
ax[1].set_title('One-step conditional return forecasts')
ax[1].set_xlim(retf_2.index[0],retf_2.index[-1])
ax[1].legend(loc='best')

# ax[2]: Mean forecast vs true
ret_true_2.plot(ax=ax[2],style='b',label='Observed returns')
retf_2.plot(ax=ax[2],style='k',label='Return forecasts')
retf_lb.plot(ax=ax[2],style='g--',label='95% IC')
retf_ub.plot(ax=ax[2],style='g--',label='')
ax[2].set_title('One-step conditional Return forecasts')
ax[2].set_xlim(retf_2.index[0],retf_2.index[-1])
ax[2].legend(loc='best')

plt.tight_layout()
plt.show()



#-------------------------------------------------------------------------------------------------------------------------------------#
#Graphs for the forecast of the second best models Q1 part C

forecast_df_3,*_ = arma_garch_estimate_and_forecast(series_1, cutoff_date, arma_order=(0,1), garch_order=(2, 2))
forecast_df_3
forecast_df_4,*_ = arma_garch_estimate_and_forecast(series_2, cutoff_date, arma_order=(2,3), garch_order=(1, 1))
forecast_df_4
fig, ax = plt.subplots(1, 3, figsize = (15,4))
ret_true_3 = forecast_df_3.loc[forecast_df_3.index>=cutoff_date,'ret_true']
retf_3 = forecast_df_3.loc[forecast_df_3.index>=cutoff_date,'retf']
varf_3 = forecast_df_3.loc[forecast_df_3.index>=cutoff_date,'varf']
retf_lb = retf_3 - 1.96*varf_3**0.5 # Lower bound 95%
retf_ub = retf_3 + 1.96*varf_3**0.5 # Upper bound 95%

# ax[0]: Conditional variance forecasts
varf_3.plot(ax=ax[0],style='k',label='Variance forecasts')
ax[0].set_title('One-step conditional Variance forecasts')
ax[0].set_xlim(varf_3.index[0],varf_3.index[-1])
ax[0].legend(loc='best')


# ax[1]: Mean forecast
retf_3.plot(ax=ax[1],style='k',label='Return forecasts')
ax[1].set_title('One-step conditional return forecasts')
ax[1].set_xlim(retf_3.index[0],retf_3.index[-1])
ax[1].legend(loc='best')

# ax[2]: Mean forecast vs true
ret_true_3.plot(ax=ax[2],style='b',label='Observed returns')
retf_3.plot(ax=ax[2],style='k',label='Return forecasts')
retf_lb.plot(ax=ax[2],style='g--',label='95% IC')
retf_ub.plot(ax=ax[2],style='g--',label='')
ax[2].set_title('One-step conditional Return forecasts')
ax[2].set_xlim(retf_3.index[0],retf_3.index[-1])
ax[2].legend(loc='best')

plt.tight_layout()
plt.show()
fig, ax = plt.subplots(1, 3, figsize = (15,4))
ret_true_4 = forecast_df_4.loc[forecast_df_4.index>=cutoff_date,'ret_true']
retf_4 = forecast_df_4.loc[forecast_df_4.index>=cutoff_date,'retf']
varf_4 = forecast_df_4.loc[forecast_df_4.index>=cutoff_date,'varf']
retf_lb = retf_4 - 1.96*varf_4**0.5 # Lower bound 95%
retf_ub = retf_4 + 1.96*varf_4**0.5 # Upper bound 95%

# ax[0]: Conditional variance forecasts
varf_4.plot(ax=ax[0],style='k',label='Variance forecasts')
ax[0].set_title('One-step conditional Variance forecasts')
ax[0].set_xlim(varf_4.index[0],varf_4.index[-1])
ax[0].legend(loc='best')


# ax[1]: Mean forecast
retf_4.plot(ax=ax[1],style='k',label='Return forecasts')
ax[1].set_title('One-step conditional return forecasts')
ax[1].set_xlim(retf_4.index[0],retf_4.index[-1])
ax[1].legend(loc='best')

# ax[2]: Mean forecast vs true
ret_true_4.plot(ax=ax[2],style='b',label='Observed returns')
retf_4.plot(ax=ax[2],style='k',label='Return forecasts')
retf_lb.plot(ax=ax[2],style='g--',label='95% IC')
retf_ub.plot(ax=ax[2],style='g--',label='')
ax[2].set_title('One-step conditional Return forecasts')
ax[2].set_xlim(retf_4.index[0],retf_4.index[-1])
ax[2].legend(loc='best')

plt.tight_layout()
plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------#
# Estimating the Root Mean Squared Prediction Error, Mean Squared Prediction Error, Root Median Squared Prediction Error related to Q1part c
def evaluate_forecasts(actual, predictions):
    # Calculate the forecast errors by subtracting actual values from predictions
    errors = predictions - actual 
    rmspe = np.sqrt(np.mean(errors**2))  # Root Mean Squared Prediction Error
    mape = np.mean(np.abs(errors))  # Mean Absolute Prediction Error
    mdape = np.median(np.abs(errors))  # Median Absolute Prediction Error
    return rmspe, mape, mdape

rmspe_1, mape_1, mdape_1 = evaluate_forecasts(ret_true_1, retf_1)
rmspe_2, mape_2, mdape_2 = evaluate_forecasts(ret_true_2, retf_2)

rmspe_3, mape_3, mdape_3 = evaluate_forecasts(ret_true_3, retf_3)# this is the secondary mnodel for spx
rmspe_4, mape_4, mdape_4 = evaluate_forecasts(ret_true_4, retf_4)#this is the secondary model for vix

print(rmspe_1, mape_1, mdape_1)
print(rmspe_2, mape_2, mdape_2)
print(rmspe_3, mape_3, mdape_3)
print(rmspe_4, mape_4, mdape_4)


# Diebold-Mario test

def diebold_mario(floss1, floss2):

    
    # Convert forecast loss inputs to numpy arrays
    floss1 = np.array(floss1)
    floss2 = np.array(floss2)
    
    # Check if the inputs have the same length
    if len(floss1) != len(floss2):
        raise ValueError('`floss1` and `floss2` must have same length.')
    # Check if the inputs have negative numbers
    if any(floss1<0) or any(floss2<0):
        raise ValueError('`floss1` and `floss2` must contain non-negative numbers.')
    
     # Calculate the loss differential between the two models
    ld = floss1 - floss2
    T = len(ld) # Number of observations
    
   # Calculate the Sign Test statistic 
    ld1 = ld > 0
    S = 2 / np.sqrt(T) * (np.sum(ld1 - 0.5))

    # Calculate DM test statistic
    DM = np.mean(ld) / np.sqrt(np.var(ld)/T)

    # Calculate p-values
    # Cumulative distribution function (CDF) of the normal distribution
    # used to to find p-value associated with normal distribution
    S_p_value = 2 * (1 - norm.cdf(np.abs(S)))
    DM_p_value = 2 * (1 - norm.cdf(np.abs(DM)))

    # Display results in a tabular format
    DM_table = pd.DataFrame({
        'Test': ['Sign Test', 'DM Test'],
        'Statistic': [S, DM],
        'P-value': [S_p_value, DM_p_value]
    })
    
    # Return the data frame with test statistics and p-values
    return DM_table

#-------------------------------------------------------------------------------------------------------------------------------------#
#Testing Differences between the two best models 


# For SPX
# Based on Squared Values
floss1 = (retf_1-ret_true_1)**2
floss3 = (retf_3-ret_true_3)**2
print(diebold_mario(floss1, floss3))


# Based on Absolute Values
floss1 = np.abs(retf_1-ret_true_1)
floss3 = np.abs(retf_3-ret_true_3)
print(diebold_mario(floss1, floss3))



#For Vix Models
floss2 = (retf_2-ret_true_2)**2
floss4 = (retf_4-ret_true_4)**2
print(diebold_mario(floss2, floss4))


# Based on Absolute Values
floss2 = np.abs(retf_2-ret_true_2)
floss4 = np.abs(retf_4-ret_true_4)
print(diebold_mario(floss2, floss4))




#-------------------------------------------------------------------------------------------------------------------------------------#
# Question 1 part D


# Helper Functions

def calculate_max_drawdown(portfolio_df):
    """
    Calculates the maximum drawdown of the portfolio.
    """
    cumulative_returns = portfolio_df['Portfolio Value']
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    return max_drawdown

def calculate_cagr(portfolio_df, initial_budget=10000):
    """
    Calculates the Compound Annual Growth Rate (CAGR) of the portfolio.
    """
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = days / 365.25
    if years > 0:
        cagr = (portfolio_df['Portfolio Value'].iloc[-1] / initial_budget) ** (1 / years) - 1
        return cagr * 100
    else:
        return np.nan

def calculate_sharpe_ratio(portfolio_df):
    """
    Calculates the Sharpe Ratio of the portfolio.
    """
    return (portfolio_df['Portfolio Return'].mean() / portfolio_df['Portfolio Return'].std()) * np.sqrt(252)

def calculate_skewness(portfolio_df):
    """
    Calculates the skewness of the portfolio returns.
    """
    return skew(portfolio_df['Portfolio Return'])


# Main Functions


def trading_strategy(forecast_df, prices, initial_budget=10000, start_date='2022-01-01', end_date='2023-10-06'):
    """
    Implements the trading strategy based on forecasted returns.
    """
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Filter forecast data for the trading period
    trading_forecasts = forecast_df.loc[start_date:end_date].copy()
    
    # Ensure 'retf_shifted' is present and drop NaNs
    if 'retf_shifted' not in trading_forecasts.columns:
        raise ValueError("'retf_shifted' column not found in forecast_df.")
    
    trading_forecasts = trading_forecasts.dropna(subset=['retf_shifted'])
    trading_dates = trading_forecasts.index.intersection(prices.index)
    
    # Initialize portfolio variables
    cash = initial_budget
    holdings = 0.0
    portfolio_history = []
    trades = []  # To record buy/sell actions
    
    for date in trading_dates:
        forecast_ret = trading_forecasts.loc[date, 'retf_shifted']
        price_today = prices.loc[date]
        
        # Decision Making
        action = 'Hold'
        shares_traded = 0.0
        
        if forecast_ret > 0 and cash > 0:
            # Buy: Invest all cash to buy shares
            shares_to_buy = cash / price_today
            holdings += shares_to_buy
            cash = 0.0
            action = 'Buy'
            shares_traded = shares_to_buy
            trades.append({'Date': date, 'Action': 'Buy', 'Shares': shares_traded, 'Price': price_today})
        elif forecast_ret < 0 and holdings > 0:
            # Sell: Liquidate all holdings
            cash += holdings * price_today
            shares_traded = holdings
            holdings = 0.0
            action = 'Sell'
            trades.append({'Date': date, 'Action': 'Sell', 'Shares': shares_traded, 'Price': price_today})
        # else:
            # Hold: No action
        
        # Calculate portfolio value
        portfolio_value = cash + holdings * price_today
        
        # Record the day's portfolio status
        portfolio_history.append({
            'Date': date,
            'Action': action,
            'Cash': cash,
            'Holdings': holdings,
            'Price': price_today,
            'Portfolio Value': portfolio_value
        })
    
    # Create DataFrames from the history
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df.set_index('Date', inplace=True)
    
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.set_index('Date', inplace=True)
    
    # Calculate 'Portfolio Return' as daily percentage change
    portfolio_df['Portfolio Return'] = portfolio_df['Portfolio Value'].pct_change().fillna(0)
    
    # Calculate metrics
    sharpe_ratio = calculate_sharpe_ratio(portfolio_df)
    portfolio_skewness = calculate_skewness(portfolio_df)
    turnover = len(trades_df) / portfolio_df['Portfolio Value'].mean() if portfolio_df['Portfolio Value'].mean() != 0 else np.nan
    
    # Compile metrics into a dictionary
    metrics = {
        'Sharpe Ratio': sharpe_ratio,
        'Skewness': portfolio_skewness,
        'Turnover': turnover
    }
    
    return portfolio_df, trades_df, metrics

def buy_and_hold(prices, start_date='2022-01-01', initial_budget=10000):
    """
    Implements the buy-and-hold strategy.
    """
    # Convert start_date to datetime
    start_date = pd.to_datetime(start_date)
    
    # Check if start_date exists in the prices index
    if start_date in prices.index:
        price_start = prices.loc[start_date]
    else:
        # Find the next available trading day after start_date
        next_dates = prices.index[prices.index > start_date]
        if len(next_dates) == 0:
            raise ValueError(f"No trading data available after {start_date}.")
        start_date = next_dates.min()
        price_start = prices.loc[start_date]
        print(f"Start date {start_date.date()} not found. Using next available trading day {start_date.date()}.")
    
    # Calculate the number of shares to buy
    holdings = initial_budget / price_start
    cash = 0.0  # All funds invested initially
    portfolio_history = []
    
    # Iterate over the trading days from start_date onwards
    for date, price in prices.loc[start_date:].items():
        portfolio_value = cash + holdings * price
        portfolio_history.append({
            'Date': date,
            'Action': 'Buy & Hold',
            'Cash': cash,
            'Holdings': holdings,
            'Price': price,
            'Portfolio Value': portfolio_value
        })
    
    # Create a DataFrame from the portfolio history
    portfolio_df = pd.DataFrame(portfolio_history)
    portfolio_df.set_index('Date', inplace=True)
    
    # Calculate 'Portfolio Return' as daily percentage change
    portfolio_df['Portfolio Return'] = portfolio_df['Portfolio Value'].pct_change().fillna(0)
    
    return portfolio_df

#-------------------------------------------------------------------------------------------------------------------------------------#
# Related to question 1 part D Visualising our models and describing them


def plot_portfolio_single(portfolio_best, portfolio_buy_hold, trades_best, series_name='SP500'):
    """
    Plots the portfolio values of the trading strategy and buy-and-hold strategy over time.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot Portfolio Values
    plt.plot(portfolio_best['Portfolio Value'], label='Best Model Strategy', color='blue')
    plt.plot(portfolio_buy_hold['Portfolio Value'], label='Buy-and-Hold Strategy', color='green', linestyle='--')
    
    # Overlay Buy/Sell Signals for Best Model
    if not trades_best.empty:
        buys_best = trades_best[trades_best['Action'] == 'Buy']
        sells_best = trades_best[trades_best['Action'] == 'Sell']
        plt.scatter(buys_best.index, portfolio_best.loc[buys_best.index, 'Portfolio Value'],
                    marker='^', color='lime', label='Buy Signal (Best Model)', s=100)
        plt.scatter(sells_best.index, portfolio_best.loc[sells_best.index, 'Portfolio Value'],
                    marker='v', color='red', label='Sell Signal (Best Model)', s=100)
        
        # Annotate number of buys and sells
        plt.text(0.02, 0.95, f"Total Buys: {buys_best.shape[0]}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.02, 0.90, f"Total Sells: {sells_best.shape[0]}", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    
    plt.title(f'Portfolio Value Over Time - {series_name}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_return_distributions(portfolio_df, buy_hold_df, series_name='SP500'):
    """
    Plots the distribution of daily returns for both strategies.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot histogram for Trading Strategy
    sns.histplot(portfolio_df['Portfolio Return'], bins=50, color='blue', alpha=0.5, label='Trading Strategy', stat='density', kde=True)
    
    # Plot histogram for Buy-and-Hold Strategy
    sns.histplot(buy_hold_df['Portfolio Return'], bins=50, color='green', alpha=0.5, label='Buy-and-Hold Strategy', stat='density', kde=True)
    
    plt.title(f'Distribution of Portfolio Returns vs Buy-and-Hold Returns - {series_name}')
    plt.xlabel('Daily Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def compile_metrics(portfolio_df, buy_hold_df, metrics_trading, metrics_buy_hold, initial_budget=10000, series_name='SP500'):
    """
    Compiles performance metrics of trading strategy and buy-and-hold strategy into a DataFrame.
    """
    # Define metrics to include
    metrics_list = ['Total Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Skewness']
    
    # Calculate metrics for Trading Strategy
    total_return_trading = (portfolio_df['Portfolio Value'].iloc[-1] / initial_budget - 1) * 100
    cagr_trading = calculate_cagr(portfolio_df, initial_budget)
    max_dd_trading = calculate_max_drawdown(portfolio_df)
    skewness_trading = calculate_skewness(portfolio_df)
    sharpe_ratio_trading = calculate_sharpe_ratio(portfolio_df)
    
    metrics_trading_complete = {
        'Total Return': total_return_trading,
        'CAGR': cagr_trading,
        'Sharpe Ratio': sharpe_ratio_trading,
        'Max Drawdown': max_dd_trading,
        'Skewness': skewness_trading
    }
    
    # Calculate metrics for Buy-and-Hold Strategy
    total_return_bh = (buy_hold_df['Portfolio Value'].iloc[-1] / initial_budget - 1) * 100
    cagr_bh = calculate_cagr(buy_hold_df, initial_budget)
    max_dd_bh = calculate_max_drawdown(buy_hold_df)
    skewness_bh = calculate_skewness(buy_hold_df)
    sharpe_ratio_bh = calculate_sharpe_ratio(buy_hold_df)
    
    metrics_buy_hold_complete = {
        'Total Return': total_return_bh,
        'CAGR': cagr_bh,
        'Sharpe Ratio': sharpe_ratio_bh,
        'Max Drawdown': max_dd_bh,
        'Skewness': skewness_bh
    }
    
    # Combine metrics into DataFrame
    metrics_data = {
        'Metric': metrics_list,
        'Trading Strategy': [metrics_trading_complete.get(metric, np.nan) for metric in metrics_list],
        'Buy-and-Hold': [metrics_buy_hold_complete.get(metric, np.nan) for metric in metrics_list]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Metric', inplace=True)
    
    # Save the metrics to CSV
    metrics_df.to_csv(f'/Users/amirhossein/Downloads/performance_metrics_{series_name}.csv')
    
    return metrics_df

def plot_comparative_bar_chart(metrics_df, series_name='SP500'):
    """
    Plots a comparative bar chart for performance metrics of both strategies.
    """
    metrics = metrics_df.index.tolist()
    trading = metrics_df['Trading Strategy'].values
    buy_hold = metrics_df['Buy-and-Hold'].values
    
    x = np.arange(len(metrics))  # label locations
    width = 0.35  # width of the bars
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, trading, width, label='Trading Strategy', color='blue')
    rects2 = ax.bar(x + width/2, buy_hold, width, label='Buy-and-Hold', color='green')
    
    # Add labels and title
    ax.set_ylabel('Value (%)')
    ax.set_title(f'Performance Metrics Comparison - {series_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    
    # Attach a text label above each bar displaying its height
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f'/Users/amirhossein/Downloads/comparative_bar_chart_{series_name}.png')
    plt.show()



def analyze_and_compare_strategies(portfolio_trading, trades_trading, portfolio_buy_hold, series_name='SP500', initial_budget=10000):
    """
    Performs a comprehensive analysis and comparison between trading strategy and buy-and-hold strategy.
    """
    # Ensure 'Portfolio Return' exists
    if 'Portfolio Return' not in portfolio_trading.columns:
        portfolio_trading['Portfolio Return'] = portfolio_trading['Portfolio Value'].pct_change().fillna(0)
    
    if 'Portfolio Return' not in portfolio_buy_hold.columns:
        portfolio_buy_hold['Portfolio Return'] = portfolio_buy_hold['Portfolio Value'].pct_change().fillna(0)
    
    # Calculate metrics for trading strategy
    sharpe_ratio_trading = calculate_sharpe_ratio(portfolio_trading)
    skewness_trading = calculate_skewness(portfolio_trading)
    turnover_trading = len(trades_trading) / portfolio_trading['Portfolio Value'].mean() if portfolio_trading['Portfolio Value'].mean() != 0 else np.nan
    metrics_trading = {
        'Sharpe Ratio': sharpe_ratio_trading,
        'Skewness': skewness_trading,
        'Turnover': turnover_trading
    }
    
    # Calculate metrics for buy-and-hold strategy
    sharpe_ratio_bh = calculate_sharpe_ratio(portfolio_buy_hold)
    skewness_bh = calculate_skewness(portfolio_buy_hold)
    metrics_buy_hold = {
        'Sharpe Ratio': sharpe_ratio_bh,
        'Skewness': skewness_bh,
        'Turnover': np.nan  # Not applicable
    }
    
    # Plot return distributions
    plot_return_distributions(portfolio_trading, portfolio_buy_hold, series_name)
    
    # Compile metrics
    metrics_df = compile_metrics(
        portfolio_df=portfolio_trading,
        buy_hold_df=portfolio_buy_hold,
        metrics_trading=metrics_trading,
        metrics_buy_hold=metrics_buy_hold,
        initial_budget=initial_budget,
        series_name=series_name
    )
    print("\nCompiled Performance Metrics DataFrame:")
    print(metrics_df)
    
    # Plot comparative bar chart
    plot_comparative_bar_chart(metrics_df, series_name)
    
    
    return metrics_df

def plot_interactive_radar_chart(metrics_df, series_name='SP500'):
    """
    Plots an interactive radar chart using Plotly.
    """
    categories = metrics_df.index.tolist()
    N = len(categories)
    
    # Append the first value to close the loop
    trading_values = metrics_df['Trading Strategy'].tolist()
    buy_hold_values = metrics_df['Buy-and-Hold'].tolist()
    trading_values += trading_values[:1]
    buy_hold_values += buy_hold_values[:1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=trading_values,
        theta=categories + [categories[0]],
        fill='toself',
        name='Trading Strategy'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=buy_hold_values,
        theta=categories + [categories[0]],
        fill='toself',
        name='Buy-and-Hold'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(0, min(trading_values + buy_hold_values)), max(trading_values + buy_hold_values)*1.1]
            )
        ),
        showlegend=True,
        title=f'Interactive Radar Chart - {series_name}'
    )
    
    # Save the radar chart as an interactive HTML file
    fig.write_html(f'/Users/amirhossein/Downloads/interactive_radar_chart_{series_name}.html')
    fig.show()



# Initialise variables
initial_budget = 10000
start_date = '2022-01-01'
end_date = '2023-10-06'



# Prepare the data
common_dates = forecast_df_1.loc[start_date:end_date].index.intersection(data_3['SP500'].loc[start_date:end_date].index)

# Filter Forecasts and Prices
filtered_forecast_df_1 = forecast_df_1.loc[common_dates]
filtered_forecast_df_1['retf_shifted'] = filtered_forecast_df_1['retf'].shift(1)
filtered_forecast_df_1 = filtered_forecast_df_1.dropna(subset=['retf_shifted'])
filtered_prices_sp500 = data_3['SP500'].loc[filtered_forecast_df_1.index]

# Verify No Missing Values
assert not filtered_forecast_df_1['retf_shifted'].isna().any(), "Missing values in 'retf_shifted'."
assert not filtered_prices_sp500.isna().any(), "Missing values in 'prices_sp500'."

# Implement Trading Strategy
portfolio_best_sp500, trades_best_sp500, metrics_best_sp500 = trading_strategy(
    forecast_df=filtered_forecast_df_1,
    prices=filtered_prices_sp500,
    initial_budget=initial_budget,
    start_date=start_date,
    end_date=end_date
)

# Implement Buy-and-Hold Strategy
portfolio_buy_hold_sp500 = buy_and_hold(
    prices=filtered_prices_sp500,
    start_date=start_date,
    initial_budget=initial_budget
)

# Plot Portfolio Performance
plot_portfolio_single(
    portfolio_best=portfolio_best_sp500,
    portfolio_buy_hold=portfolio_buy_hold_sp500,
    trades_best=trades_best_sp500,
    series_name='SP500'
)

print("\n=== Performance Metrics ===")
print("Best Model (SP500):", metrics_best_sp500)

# Calculate and Print Additional Metrics for Best Model
total_return_best = (portfolio_best_sp500['Portfolio Value'].iloc[-1] / initial_budget - 1) * 100
cagr_best = calculate_cagr(portfolio_best_sp500, initial_budget)
max_dd_best = calculate_max_drawdown(portfolio_best_sp500)

print(f"\nBest Model - Total Return: {total_return_best:.2f}%")
print(f"Best Model - CAGR: {cagr_best:.2f}%")
print(f"Best Model - Max Drawdown: {max_dd_best:.2f}%")
print(f"Best Model - Skewness: {metrics_best_sp500['Skewness']:.4f}")

# Update metrics_best_sp500 with additional metrics
metrics_best_sp500.update({
    'Total Return': total_return_best,
    'CAGR': cagr_best,
    'Max Drawdown': max_dd_best
})

# Calculate Metrics for Buy-and-Hold Strategy
metrics_buy_hold_sp500 = {
    'Sharpe Ratio': calculate_sharpe_ratio(portfolio_buy_hold_sp500),
    'Skewness': calculate_skewness(portfolio_buy_hold_sp500),
    'Turnover': np.nan  # Not applicable for Buy-and-Hold
}
def plot_returns_correlation(portfolio_trading, portfolio_buy_hold, series_name='SP500'):
    """
    Plots a scatter plot to visualize the correlation between trading strategy and buy-and-hold returns.
    """
    # Merge the two DataFrames on their indices (dates)
    merged_df = pd.merge(
        portfolio_trading[['Portfolio Return']], 
        portfolio_buy_hold[['Portfolio Return']], 
        left_index=True, 
        right_index=True, 
        how='inner',
        suffixes=('_trading', '_buy_hold')
    )
    
    # Drop any rows with NaN values
    merged_df.dropna(inplace=True)
    
    # Set the plot size
    plt.figure(figsize=(8, 6))
    
    # Create scatter plot
    sns.regplot(
        x='Portfolio Return_trading', 
        y='Portfolio Return_buy_hold', 
        data=merged_df, 
        scatter_kws={'alpha':0.5},
        line_kws={'color':'red'}
    )
    


# Analyze and Compare Strategies
metrics_df_sp500 = analyze_and_compare_strategies(
    portfolio_trading=portfolio_best_sp500,
    trades_trading=trades_best_sp500,
    portfolio_buy_hold=portfolio_buy_hold_sp500,
    series_name='SP500',
    initial_budget=initial_budget
)

# Plot Interactive Radar Chart
plot_interactive_radar_chart(metrics_df_sp500, series_name='SP500')

# Computing Correlation between the two strategies 

def compute_strategy_buyhold_correlation(portfolio_trading, portfolio_buy_hold):
    """
    Computes the Pearson correlation between daily returns of the trading strategy and the buy-and-hold strategy.
    
    """
    # Ensure both DataFrames have 'Portfolio Return' column
    if 'Portfolio Return' not in portfolio_trading.columns:
        raise ValueError("'Portfolio Return' column not found in portfolio_trading DataFrame.")
    if 'Portfolio Return' not in portfolio_buy_hold.columns:
        raise ValueError("'Portfolio Return' column not found in portfolio_buy_hold DataFrame.")
    
    # Align the two DataFrames on their indices (dates)
    merged_df = pd.merge(
        portfolio_trading[['Portfolio Return']], 
        portfolio_buy_hold[['Portfolio Return']], 
        left_index=True, 
        right_index=True, 
        how='inner',
        suffixes=('_trading', '_buy_hold')
    )
    
    # Drop any rows with NaN values to ensure accurate correlation
    merged_df.dropna(inplace=True)
    
    # Compute the Pearson correlation
    correlation = merged_df['Portfolio Return_trading'].corr(merged_df['Portfolio Return_buy_hold'])
    
    return correlation

# Compute the correlation between Trading Strategy and Buy-and-Hold Strategy returns
correlation = compute_strategy_buyhold_correlation(
    portfolio_trading=portfolio_best_sp500,
    portfolio_buy_hold=portfolio_buy_hold_sp500
)

# Display the correlation result
print(f"Correlation between Trading Strategy Returns and Buy-and-Hold Returns: {correlation:.4f}")





#End of Q1
#-------------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------------------------#
#Q2
# In[5]:

warnings.simplefilter('ignore', ValueWarning)


# In[6]:


data = pd.read_csv(data_1_link)
data['DATE'] = pd.to_datetime(data['DATE'], format='%d/%m/%Y')
data.set_index('DATE', inplace=True)
data.head()


# In[14]:


#select columns
columns = ['SP500','EUROSTOXX50','FTSE100', 'DAX30']
group = data[columns].dropna()


descriptive_stats = group.describe()

# Calculate median
median = group.median()

# Calculate skewness
skewness = group.skew()

# Calculate the kurtosis 
kurtosis = group.kurtosis() + 3 # the .kurtosis() function compute excess kurtosis, i.e. Kurtosis-3

# Append calculated skewness and kurtosis values to the existing descriptive statistics DataFrame
descriptive_stats.loc['Skewness'] = skewness
descriptive_stats.loc['Kurtosis'] = kurtosis

# Perform Jarque-Bera test to assess normality 
jb_stat, jb_pvalue = jarque_bera(data) 

# Append the Jarque-Bera test statistic and its p-value to the descriptive statistics DataFrame
descriptive_stats.loc['Jarque-Bera'] = jb_stat
descriptive_stats.loc['JB pvalue'] = jb_pvalue

# Display the updated descriptive statistics
print(descriptive_stats)
descriptive_stats.to_csv('/Users/amirhossein/Downloads/Desc_stats_2.csv')

# In[15]:


group.plot(figsize=(14, 7))  # Set the figure size 
plt.title('Stock Returns /time')  # Add a title to the plot
plt.xlabel('Date')  # Label the x-axis as 'Date'
plt.ylabel('Return')  # Label the y-axis as 'Yield'
plt.legend()  # Display a legend to identify the different yield series
plt.grid(True)  # Enable grid lines 
plt.show()  


# In[9]:


# Define a function to perform the Augmented Dickey-Fuller (ADF) Test to check for stationarity in a time series
def adf_test(series, name):
    print(f'Performing ADF Test on {name}')  
    # The ADF test is applied to the series after dropping any missing values
    # 'maxlag=20' sets the maximum number of lags to the test
    result = adfuller(series.dropna(), maxlag=20)
    print('ADF Statistic: %f' % result[0])  # Display the ADF test statistic
    print('p-value: %f' % result[1])  # Display the p-value of the test
    print('Critical Values:')  # Display the critical values for different significance levels
    for key, value in result[4].items():  # Iterate over the dictionary of test statistic values
        print(f'\t{key}: {value}')  # Print each critical value
    print()  # Print a newline for better readability between outputs
    
# Apply the ADF test to each column specified in the DataFrame `group_small`
for column in group:
    if column in group.columns:  # Check if the column exists in the DataFrame to avoid key errors
        adf_test(group[column], column)  # Call the adf_test function for each column
    else:
        print(f"{column} is not in the dataset")  # Notify if a column is not found in the DataFrame


# In[10]:


#Null hypothesis of unit roots not rejected for any stock as p>0.05, impying all contain unit roots (are integrated)
#Now to determine order of intragratedness
# Loop through each column in the DataFrame `group_small` to apply the Augmented Dickey-Fuller (ADF) test
for column in group:
    # Check if the column exists within the DataFrame to avoid key errors
    if column in group.columns:
        # Calculate the first difference of the series using  .diff()
        #`f'd({column})'` formats the name to indicate the difference (e.g., d(YIELD3M)) for clarity in output
        adf_test(group[column].diff(), f'd({column})')
    else:
        # Output an error message if the column is not found in the DataFrame
        print(f"{column} is not in the dataset")


# In[ ]:


#null hyptoehsis of unit roots rejected for all first differences, therefore all series integreated of order 1 / I(1)


# In[17]:


def johansen_test(dat, det_order=0, k_ar_diff=1, signif=0.05):
    # Perform the Johansen Cointegration test using specified parameters
    # 'det_order' refers to deterministic terms: 0 = no deterministic term, -1 = constant term, 1 = linear trend
    # 'k_ar_diff' specifies the number of lags minus one (the lag order)
    result = coint_johansen(dat, det_order, k_ar_diff)
    
    # Trace Statistics
    trace_df = pd.DataFrame()  # Initialize a DataFrame to hold Trace test results
    # Generate the hypothesis test descriptions for the number of cointegrating equations (CEs)
    trace_df['No. of CE(s)'] = ['None'] + ['At most ' + str(x) for x in range(1, len(dat.columns))]
    trace_df['Eigenvalue'] = result.eig  # Eigenvalues from the Johansen test
    trace_df['Trace Stat'] = result.trace_stat  # Trace statistics (or likelihood ratio stats)
    signifvec = np.array([0.1, 0.05, 0.01])  # Pre-defined significance levels
    # Select the critical values that match the user-specified significance level
    trace_df[str(signif) + ' Crit Value'] = result.trace_stat_crit_vals[:, signifvec == signif]
    
    # Max-Eigen Statistics
    maxeig_df = pd.DataFrame()  # Initialize a DataFrame to hold Max-Eigen test results
    maxeig_df['No. of CE(s)'] = trace_df['No. of CE(s)']  # Same hypothesis test descriptions as for Trace test
    maxeig_df['Eigenvalue'] = trace_df['Eigenvalue']  # Eigenvalues from the Johansen test
    maxeig_df['Max-Eigen Stat'] = result.max_eig_stat  # Maximum eigenvalue statistics
    # Select the critical values that match the user-specified significance level for Max-Eigen test
    maxeig_df[str(signif) + ' Crit Value'] = result.max_eig_stat_crit_vals[:, signifvec == signif]
    
    # Output the results
    print("Johansen's Cointegration Trace test")
    print(trace_df) 
    print("Johansen's Cointegration Max-Eigen test")
    print(maxeig_df) 


# In[18]:


#Applying Johansen_procedure, interpret results for VEC model for series 
johansen_test(group)


# In[19]:


# Select the optimal number of lags <=5
# 'group' is the DataFrame containing the time series data
# 'maxlags=5' specifies the maximum number of lags to consider during the lag order selection process
# in practise we often set the maximum lags to be 5 to avoid overfitting 
# deterministic' set to 'ci' includes a constant and a trend in the model as part of the lag selection criteria

lag_selection = select_order(group, maxlags=5, deterministic='ci')
lag_selection.summary()


# In[20]:


lag_selection.selected_orders
#resuls point towards lag of 5


# In[21]:


model = VECM(group, k_ar_diff=2, coint_rank=3, deterministic='ci')

# Fit the model to the data 
vecm_result = model.fit()

# Output the summary of the VECM fit
print("VECM Summary:")
vecm_result.summary()


# In[ ]:


# Showing the results from the Ganger Causality test 



print(vecm_result.test_granger_causality(caused = 'SP500', causing='EUROSTOXX50', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'EUROSTOXX50', causing='SP500', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'SP500', causing='FTSE100', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'FTSE100', causing='SP500', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'SP500', causing='DAX30', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'DAX30', causing='SP500', signif=0.05).summary())

print(vecm_result.test_granger_causality(caused = 'EUROSTOXX50', causing='FTSE100', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'FTSE100', causing='EUROSTOXX50', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'EUROSTOXX50', causing='DAX30', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'DAX30', causing='EUROSTOXX50', signif=0.05).summary())

print(vecm_result.test_granger_causality(caused = 'FTSE100', causing='DAX30', signif=0.05).summary())
print(vecm_result.test_granger_causality(caused = 'DAX30', causing='FTSE100', signif=0.05).summary())


# In[43]:
# Showing the summary of the results 




# In[44]:


_ = vecm_result.irf().plot()


# In[47]:


VARmodel = VAR(group)
lag_selection = VARmodel.select_order(maxlags=5)
lag_selection.summary()


# In[48]:


lag_selection.selected_orders


# In[52]:


#suggests 5 lags again
var_result = VARmodel.fit(5)
print(var_result.summary())
var_result.summary()


# In[55]:
# Showing the Results of the tests

print(var_result.test_causality(caused = 'SP500', causing='EUROSTOXX50', signif=0.05).summary())
print(var_result.test_causality(caused = 'EUROSTOXX50', causing='SP500', signif=0.05).summary())
print(var_result.test_causality(caused = 'SP500', causing='FTSE100', signif=0.05).summary())
print(var_result.test_causality(caused = 'FTSE100', causing='SP500', signif=0.05).summary())
print(var_result.test_causality(caused = 'SP500', causing='DAX30', signif=0.05).summary())
print(var_result.test_causality(caused = 'DAX30', causing='SP500', signif=0.05).summary())

print(var_result.test_causality(caused = 'EUROSTOXX50', causing='FTSE100', signif=0.05).summary())
print(var_result.test_causality(caused = 'FTSE100', causing='EUROSTOXX50', signif=0.05).summary())
print(var_result.test_causality(caused = 'EUROSTOXX50', causing='DAX30', signif=0.05).summary())
print(var_result.test_causality(caused = 'DAX30', causing='EUROSTOXX50', signif=0.05).summary())

print(var_result.test_causality(caused = 'FTSE100', causing='DAX30', signif=0.05).summary())
print(var_result.test_causality(caused = 'DAX30', causing='FTSE100', signif=0.05).summary())


# In[56]:
# Ploting the causalities 

__ = var_result.irf().plot()
plt.show()

# In[ ]:



# Calculate the correlation matrix
corr_matrix = group.corr()

# Set up the matplotlib figure
plt.figure(figsize=(8, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap
sns.heatmap(
    corr_matrix, 
    annot=True,        
    cmap='coolwarm',   
    fmt=".2f",         
    linewidths=.5,     
    square=True,       
    cbar_kws={"shrink": .75}  
)

# Add title and labels
plt.title('Correlation Heatmap - SP500 and Other Indices', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal

# Improve layout and display the plot
plt.tight_layout()
plt.show()
