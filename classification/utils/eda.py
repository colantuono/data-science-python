import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scis
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


def frequency_table(dataset: pd.DataFrame, column: str):
    return pd.concat([dataset[column].value_counts().rename('Counts'),
                     dataset[column].value_counts(normalize=True).rename('Frequency')], axis = 1)
    
## Data Transformation 
def log_transform(x):
    desc = x.describe()
    if desc['min'] == 0: shift = 0.01;
    else: shift = 0
    ln= np.log(x+shift) 
    return ln

# Power of 3 transformation
def power_of_3(x):
    return (x)**(1/3)

# Power of 3 transformation
def power_of_4(x):
    return (x)**(1/4)

# Square root transformation
def sqrt_transform(x):
    return np.sqrt(x)

# Exponential transformation with cap
def exp_transform(x, cap=20):
    capped_x = np.minimum(x, cap)
    return np.exp(capped_x)

# Z-Score transformation
def robust_transform(x):
    mean = x.mean()
    iqr = np.subtract(*np.percentile(x, [75,25]))
    return (x-mean) / iqr

    
def transform_and_plot(data, bins=25):
    sns.set_theme()
    for column in data.columns:
        x = data[column] + 0.001
        ln = log_transform(x)
        pw3 = power_of_3(x)
        pw4 = power_of_4(x)
        sqrt = sqrt_transform(x)
        rt = robust_transform(x)
        
        
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(21,7))
        ax[0 , 0].hist(x , bins=bins)
        ax[0 , 0].set_title(f'{column} - Original (skew = {x.skew():.4f})')
        ax[0 , 1].hist(ln , bins=bins)
        ax[0 , 1].set_title(f'{column} - Log (skew = {ln.skew():.4f})')
        ax[0 , 2].hist(pw3 , bins=bins)
        ax[0 , 2].set_title(f'{column} - Power of 3 (skew = {pw3.skew():.4f})')
        ax[1 , 0].hist(pw4 , bins=bins)
        ax[1 , 0].set_title(f'{column} - Power of 4 (skew = {pw4.skew():.4f})') 
        ax[1 , 1].hist(sqrt, bins=bins)
        ax[1 , 1].set_title(f'{column} - Square Root (skew = {sqrt.skew():.4f})')
        ax[1 , 2].hist(rt, bins=bins)
        ax[1 , 2].set_title(f'{column} - Robust Scalling (skew = {rt.skew():.4f})') 
        plt.show();
    
def transformed_data_skew(dataset):
    skew = pd.DataFrame(index = ['x','ln','pw3','pw4','sqrt','rt'])
    for i, column in enumerate(dataset.columns):
        x = dataset[column] + 0.001
        ln = log_transform(x)
        pw3 = power_of_3(x)
        pw4 = power_of_4(x)
        sqrt = sqrt_transform(x)
        rt = robust_transform(x)
        
        x = x.skew()
        ln = ln.skew()
        pw3 = pw3.skew()
        pw4 = pw4.skew()
        sqrt = sqrt.skew()
        rt = rt.skew()
    
        skew[column] = [x,ln,pw3,pw4,sqrt,rt]
    return skew



def custom_boxplot(dataset, columns, rows, cols, suptitle=None ):
    fig, axs = plt.subplots(rows, cols, sharey=True, figsize=(4.5*cols,3.5*rows))
    fig.suptitle(suptitle, y=1, size=20)
    axs=axs.flatten()
    
    for i, column in enumerate(columns):
        sns.boxplot(data=dataset[column], orient='h', ax=axs[i])
        axs[i].set_title(f'{column}, skewness: {round(dataset[column].skew(axis = 0, skipna= True), 2)}', size=12)
    
    fig.subplots_adjust(hspace=0.5, wspace=0.5)    

def outliers_filter(vector, output_outliers=False):
    vector=np.array(vector)
    q1 = np.quantile(vector, 0.25)
    q3 = np.quantile(vector, 0.75)
    iqr = q3 - q1
    lower_lim = q1 - 1.5 * iqr
    upper_lim = q3 + 1.5 * iqr
    qty_outl = ((vector < lower_lim) | (vector > upper_lim)).sum()
    print(f'Observations: {len(vector)}')
    print(f'Outliers: {qty_outl}, {qty_outl/len(vector):.2%} ')
    print(f'Interquartile Distance: {iqr:.4f}')
    print(f'Lower Limit, Upper Limit: [{lower_lim:.2f}, {upper_lim:.2f}]')
    print(f'Minimum, Maximum: [{np.min(vector):.2f}, {np.max(vector):.2f}]')
    if output_outliers:
        vector = vector[(np.where(vector > lower_lim))]
        vector = vector[(np.where(vector < upper_lim))]
        return vector
        
        
def numeric_correlation(dataset, columns, method, limit=0.7):
    corr_matrix = dataset[columns].corr(method=method)
    corr_lim = limit
    rng = np.arange(len(corr_matrix))
    s_corr = corr_matrix.mask(rng[:,None] <= rng).stack()
    df_corr = pd.DataFrame(s_corr, columns=[f'{method}_Correlation']).reset_index()
    df_corr = df_corr[df_corr[f'{method}_Correlation']> corr_lim]
    df_corr.sort_values(by=f'{method}_Correlation', ascending=False)
    return df_corr

def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtypes.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d0 = d0.astype({"x": str})
        d = d0.groupby("x", as_index=False, dropna=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Non-Events']/d['% of Events'])
        d['IV'] = d['WoE'] * (d['% of Non-Events']-d['% of Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF


def predictive_power(x):
    if x < 0.02: return 'useless'
    elif 0.02 <= x <= 0.1 : return 'weak'
    elif 0.1 <= x <= 0.3 : return 'medium'
    elif 0.3 <= x <= 0.5 : return 'strong'
    elif x > 0. : return 'suspicious'
    
    
    