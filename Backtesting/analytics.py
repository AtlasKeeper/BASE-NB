import pandas as pd
import os
from datetime import datetime 
import datetime as dt 
import numpy as np
import time
import glob
# from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import shutil
import multiprocessing
from tqdm import tqdm


###############################################  FINAL TRADESHEET CREATOR  ###########################################
# INPUT FILES ARE:
# filter_df6
# Trade_Sheets folder

####################################################################################################
stock = 'NIFTY'
# stock = 'FINNIFTY'
# stock = 'FINNIFTY'                        
# option_type = 'CE'
option_type = 'ND'

superset = 'SHORT_STRADDLE_rajesh_sir/plain_vanilla'
# superset = 'EMA Crossover'
# superset = 'EMA MACD Crossover'
# superset = 'EMA MACD Support'
# superset = 'RSI'

# root_dir = f"/home/newberry/{superset} copy/"
root_dir = f"/home/newberry4/jay_test/{superset}/"
# filter_df = pd.read_csv(root_dir + f'temp_data/{stock}/{option_type}/filter_df6.csv')
tradesheet_folder = root_dir + f'/{stock}/{option_type}/Trade_Sheets/'
output_folder = root_dir + f'/{stock}/{option_type}/final_tradesheet/'


####################################################################################################

os.makedirs(output_folder, exist_ok=True)

lot_size_dict = {'NIFTY': 75,'FINNIFTY': 25,
            'BANKNIFTY': 15 , 'SENSEX' : 20}
govt_tc_dict = {"NIFTY": 2.25, 'FINNIFTY': 2.25 ,
           "BANKNIFTY": 3 , 'SENSEX' : 3}

Date = {'start': '2021-06-01', 'end': '2024-11-30'}
total_months = 36



########################################### daily pnl ##################################


inputfolder_path = root_dir + f'/{stock}/{option_type}/Trade_Sheets/'
outputfolder_path = root_dir + f'/{stock}/{option_type}/dailypnl/'
#####################################################################################################

files = os.listdir(inputfolder_path)
dfs = {}


def get_spread_from_range(value, stock, lot_size):
    if stock=='NIFTY' or stock=='FINNIFTY':
        range_dict = {'(0, 10)': 0.05,
                      '(10.001, 33)': 0.1
                     }
        
    elif stock=='BANKNIFTY':
        range_dict = {'(0, 10)': 0.05,
                      '(10.001, 33)': 0.1
                     }
    elif stock=='SENSEX':
        range_dict = {'(0, 10)': 0.05,
                      '(10.001, 33)': 0.1
                     }
          
    for key, range_tuple in range_dict.items():
        start, end = eval(key)
        if start <= abs(value) <= end:
            return abs(range_tuple * lot_size)
       
    return (abs(value * lot_size * 0.3) / 100)



for file in files:
    if file.endswith('.csv') or file.endswith('.xls'):
        file_path = os.path.join(inputfolder_path, file)
        #print(file_path)
        df = pd.read_csv(file_path)
        dfs[file] = df

for file, df in dfs.items():
    # if file.startswith('NIFTY'):
    if file.startswith('NIFTY'):
        idx_calc = 'NIFTY'
    elif file.startswith("BANKNIFTY"):
        idx_calc = 'BANKNIFTY'
    elif file.startswith("FINNIFTY"):
        idx_calc = 'FINNIFTY'
    elif file.startswith("SENSEX"):
        idx_calc = 'SENSEX'
    
    govt_charge = govt_tc_dict[idx_calc]
    lot_size = lot_size_dict[idx_calc]

    pnl_list = []
    for idx, row in df.iterrows():
        row_premium = row['Premium']
        row_tc = abs(get_spread_from_range(row_premium, idx_calc, lot_size)) + govt_charge 
        row_pnl = (row_premium * lot_size) - (row_tc)
        pnl_list.append(row_pnl)

    df['PnL'] = pnl_list

    result_df = df.groupby(['Date']).agg({
        'ExpiryDate': 'first',
        'PnL': 'sum'
    }).reset_index()

    result_df.sort_values(by='Date', inplace=True)

    if not os.path.exists(outputfolder_path):
        os.makedirs(outputfolder_path)
    

    # Save the result_df as an Excel file in the 'dailypnl/' folder
    output_file_path = os.path.join(outputfolder_path, f'{file}')

    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exists. Skipping.")
        continue

    selected_columns = ['Date', 'ExpiryDate', 'PnL']
    result_df[selected_columns].to_csv(output_file_path, index=False)


########################################################################################################################3
#filtering according to each period profitability



import pandas as pd     
import os   
existing_df = pd.DataFrame() 
#tradesheet_df = pd.read_csv('/home/newberry2/vix2_analytics_data/PNL_DTE/dte-0/tradesheet_dte0.csv')
#pnl_df = pd.read_csv('/home/newberry2/vix2_analytics_data/PNL_DTE/dte-0/pnl_dte0_dtecol.csv')
#output_folder = "/home/newberry2/vix2_analytics_report/"


def minPnl(Date, df):
    monthly31 = {'start': '2021-06-01', 'end': '2024-11-30'}
    monthly11 = {'start': '2023-11-01', 'end': '2024-11-30'}
    monthly3 = {'start': '2024-08-01', 'end': '2024-11-30'}


    def calculate_monthly_pnl(Date, df):
        start_date = Date['start']
        end_date = Date['end']

        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        monthly_pnls = df.loc[mask, 'PnL'].tolist()

        return sum(monthly_pnls)  # Sum the PnL values instead of returning the list

    # Pass the 'days_to_expiry' argument when calling calculate_monthly_pnl
    Dmonthly31_pnl = calculate_monthly_pnl(monthly31, df) / 40 
    Dmonthly11_pnl = calculate_monthly_pnl(monthly11, df) / 12 
    Dmonthly3_pnl = calculate_monthly_pnl(monthly3, df) / 4  
    all_monthly_pnls = [Dmonthly31_pnl, Dmonthly11_pnl, Dmonthly3_pnl]

    # Remove None values before finding the overall minimum
    all_pnls = [pnl for pnl in all_monthly_pnls if pnl is not None]
    # overall_min_pnl = min(all_pnls, default=None)
    overall_min_pnl = Dmonthly31_pnl

    return overall_min_pnl,Dmonthly31_pnl,Dmonthly11_pnl,Dmonthly3_pnl

# Drawdown
def get_drawdown(Date, PnL):
    
    max_drawdown = 0
    max_drawdown_percentage = 0
    max_drawdown_date = None
    time_to_recover = 0
    peak_date_before_max_drawdown = None
    
    cum_pnl = 0
    peak = 0
    peak_date = Date.iloc[0]
    # peak_date = dt.datetime.strptime(Date[0], '%Y-%m-%d')
    
    for date, pnl in zip(Date, PnL):
        cum_pnl += pnl
        if (time_to_recover is None) and (cum_pnl >= peak):
            time_to_recover = (date - peak_date).days
            # time_to_recover = (dt.datetime.strptime(date, '%Y-%m-%d') - peak_date).days
            
        if cum_pnl >= peak:
            peak = cum_pnl
            peak_date = date
            # peak_date = dt.datetime.strptime(date, '%Y-%m-%d')
        
        drawdown = peak - cum_pnl
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            if peak != 0:
                max_drawdown_percentage = 100*max_drawdown/peak
            max_drawdown_date = date
            peak_date_before_max_drawdown = peak_date
            time_to_recover = None
    
    return max_drawdown, max_drawdown_percentage, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown


def analytics(sub_dataframe_dailypnl, existing_df, filename_str, max_investment):
    total_months = 40
    sub_dataframe_dailypnl['PnL'] = pd.to_numeric(sub_dataframe_dailypnl['PnL'], errors='coerce')

    # Convert 'Date' column to datetime format if not already in datetime
    sub_dataframe_dailypnl['Date'] = pd.to_datetime(sub_dataframe_dailypnl['Date'], errors='coerce')

    totalpnl = sub_dataframe_dailypnl['PnL'].sum()

    # Call get_drawdown function for overall period
    max_drawdown, max_drawdown_percentage, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown = get_drawdown(
        sub_dataframe_dailypnl['Date'],
        sub_dataframe_dailypnl['PnL']
    )

    
    overall_min_pnl, Dmonthly31_pnl, Dmonthly11_pnl, Dmonthly3_pnl = minPnl(sub_dataframe_dailypnl['Date'], sub_dataframe_dailypnl)

    if Dmonthly11_pnl is None or pd.isna(Dmonthly11_pnl):
        print("Warning: Dmonthly11_pnl is missing or invalid.")
        Dmonthly11_pnl = 0 

    # Additional operations
    Profits = sub_dataframe_dailypnl[sub_dataframe_dailypnl['PnL'] > 0]['PnL']
    Losses = sub_dataframe_dailypnl[sub_dataframe_dailypnl['PnL'] <= 0]['PnL']

    total_trades = len(sub_dataframe_dailypnl)
    num_winners = len(Profits)
    num_losers = len(Losses)
    win_percentage = 100 * num_winners / total_trades
    loss_percentage = 100 * num_losers / total_trades

    max_profit = Profits.max() if num_winners > 0 else 0
    max_loss = Losses.min() if num_losers > 0 else 0

    median_pnl = sub_dataframe_dailypnl['PnL'].median()
    median_profit = Profits.median() if num_winners > 0 else 0
    median_loss = Losses.median() if num_losers > 0 else 0

    sd_pnl = sub_dataframe_dailypnl['PnL'].std()
    sd_profit = Profits.std() if num_winners > 0 else 0
    sd_loss = Losses.std() if num_losers > 0 else 0

    # max_investment is already provided

    roi_with_dd = 100 * (totalpnl) / (max_investment + max_drawdown)
    roi = 100 * overall_min_pnl * total_months / max_investment

    # Calculate drawdown for the last 15 months from the date range
    date_range_start = pd.to_datetime('2023-06-01')
    date_range_end = pd.to_datetime('2024-09-30')

    # Filter the data for the last 15 months
    filtered_data = sub_dataframe_dailypnl[(sub_dataframe_dailypnl['Date'] >= date_range_start) & (sub_dataframe_dailypnl['Date'] <= date_range_end)]

    # Calculate the drawdown for the filtered data
    last_15_months_max_drawdown, last_15_months_max_drawdown_percentage, last_15_months_max_drawdown_date, last_15_months_time_to_recover, last_15_months_peak_date_before_max_drawdown = get_drawdown(
        filtered_data['Date'],
        filtered_data['PnL']
    )
    
    # Calculate Monthly Return and Sortino Ratio
    monthly_return = totalpnl / ( max_investment * total_months )
    minimum_return_needed = 0.003
    sd_loss_ratio = sd_loss/max_investment
    sortino_ratio = (monthly_return - minimum_return_needed) / sd_loss_ratio if sd_loss > 0 else 0

    print('x')
    # Create a DataFrame with the metrics
    result_df = pd.DataFrame({
        'Filename': [filename_str],
        'Total PnL': [totalpnl],
        'Max Drawdown': [max_drawdown],
        'Max Drawdown Percentage': [max_drawdown_percentage],
        '31M Monthly PnL': [Dmonthly31_pnl],
        '11M Monthly PnL': [Dmonthly11_pnl],
        'Daily 3M PnL ': [Dmonthly3_pnl],
        'min pnl ': [overall_min_pnl],
        'Max Investment': [max_investment],
        'ROI % ': [roi],
        'ROI with DD': [roi_with_dd],
        'Max Drawdown Date': [max_drawdown_date],
        'Time to Recover': [time_to_recover],
        'Peak Date Before Max Drawdown': [peak_date_before_max_drawdown],
        'Total Trades': [total_trades],
        'No. of Winners': [num_winners],
        'No. of Losers': [num_losers],
        'Win %': [win_percentage],
        'Loss %': [loss_percentage],
        'Max Profit': [max_profit],
        'Max Loss': [max_loss],
        'Median PnL': [median_pnl],
        'Median Profit': [median_profit],
        'Median Loss': [median_loss],
        'SD': [sd_pnl],
        'SD Profit': [sd_profit],
        'SD Loss': [sd_loss],
        'Last 15 Months Max Drawdown': [last_15_months_max_drawdown],
        'Last 15 Months Max Drawdown Percentage': [last_15_months_max_drawdown_percentage],
        'Last 15 Months Max Drawdown Date': [last_15_months_max_drawdown_date],
        'Last 15 Months Time to Recover': [last_15_months_time_to_recover],
        'Last 15 Months Peak Date Before Max Drawdown': [last_15_months_peak_date_before_max_drawdown],
        'Sortino Ratio': [sortino_ratio]
    })

    
    # Drop strategies where total PnL, Dmonthly11_pnl, or Sortino Ratio is negative
    # if totalpnl < 0 or Dmonthly11_pnl < 0 or sortino_ratio <= 0:
    #     return existing_df  # Skip appending this row if the conditions are met
    
    # Append the result to the existing dataframe
    existing_df = pd.concat([existing_df, result_df], ignore_index=True)


    return existing_df





# alldte_pnl_files = '/home/newberry4/jay_test/Vix_strategy/NIFTY/ND/dailypnl/'
alldte_pnl_files = root_dir + f'/{stock}/{option_type}/dailypnl/'
if stock == 'SENSEX':
    base_investment = 300000            # sensex
if stock =='NIFTY':
    base_investment = 350000     #nifty
if stock == 'BANKNIFTY':
    base_investment =  250000              #bnf
if stock == 'FINNIFTY':
    base_investment =  260000                #finnifty
tradesheet_folder = root_dir + f'/{stock}/{option_type}/Trade_Sheets/'


# Function to extract the number of lots from the filename


for root, dirs, files in os.walk(alldte_pnl_files):
    for file in files:
        file_path = os.path.join(root, file)
        print("hi")
        
        # Load the file into a DataFrame
        # pnl_df = pd.read_excel(file_path)
        pnl_df = pd.read_csv(file_path)
        # Check if the DataFrame is empty
        if pnl_df.empty:
            print(f"Skipping empty file: {file}")
            continue  # Skip the rest of the loop if the file is empty
        
        # Extract the number of lots from the filename
        # lots = extract_lots_from_filename(file)
        
        # Calculate max_investment based on the number of lots
        # if lots:
        #     max_investment = base_investment_per_lot * lots
        # else:
        #     print(f"Could not determine lots for file {file}, skipping.")
        #     continue

        # Process the file if it is not empty
        filename_str = str(file)
        print(filename_str)

        # if filename_str == 'gap_up_gap_down_intraday_lot_2_1.csv' :
        #     max_investment = base_investment * 2
        # elif filename_str == 'gap_up_gap_down_intraday_lot_1_1.csv' :
        #     max_investment = base_investment
        # elif filename_str == 'gap_up_gap_down_intraday_lot_3_1.csv' :
        #     max_investment = base_investment * 3
        # elif filename_str == 'gap_up_gap_down_intraday_lot_3_2.csv' :
        #     max_investment = base_investment * 3
        # else :
        max_investment = base_investment

        full_file_path_tradesheet = os.path.join(tradesheet_folder, filename_str)

        if os.path.isfile(full_file_path_tradesheet):
                sub_dataframe_tradesheet = pd.read_csv(full_file_path_tradesheet)

        # max_investment = ((sub_dataframe_tradesheet['Ce_Short_Atm_En_Price'].max()) * lot_size)

        print(max_investment)
        existing_df = analytics(pnl_df, existing_df, filename_str, max_investment)




from scipy.stats import zscore

def compute_final_z_scores(final_df):
    # Calculate ratios and z-scores for each required metric
    final_df['Dmonthly11_PnL_Ratio'] = final_df['11M Monthly PnL'] / final_df['Max Investment']
    final_df['Max_Drawdown_Ratio'] = final_df['Max Drawdown'] / final_df['Max Investment']
    final_df['Win %'] = final_df['Win %']
    final_df['Sortino Ratio'] = final_df['Sortino Ratio']

    # Calculate z-scores for each metric
    final_df['Dmonthly11_PnL_Z'] = zscore(final_df['Dmonthly11_PnL_Ratio'])
    final_df['Max_Drawdown_Z'] = zscore(final_df['Max_Drawdown_Ratio'])
    final_df['Win%_Z'] = zscore(final_df['Win %'])
    final_df['Sortino_Ratio_Z'] = zscore(final_df['Sortino Ratio'])

    # Standardize z-scores to range [0, 1]
    final_df['Dmonthly11_PnL_Z'] = (final_df['Dmonthly11_PnL_Z'] - final_df['Dmonthly11_PnL_Z'].min()) / (final_df['Dmonthly11_PnL_Z'].max() - final_df['Dmonthly11_PnL_Z'].min())
    final_df['Max_Drawdown_Z'] = (final_df['Max_Drawdown_Z'] - final_df['Max_Drawdown_Z'].min()) / (final_df['Max_Drawdown_Z'].max() - final_df['Max_Drawdown_Z'].min())
    final_df['Win%_Z'] = (final_df['Win%_Z'] - final_df['Win%_Z'].min()) / (final_df['Win%_Z'].max() - final_df['Win%_Z'].min())
    final_df['Sortino_Ratio_Z'] = (final_df['Sortino_Ratio_Z'] - final_df['Sortino_Ratio_Z'].min()) / (final_df['Sortino_Ratio_Z'].max() - final_df['Sortino_Ratio_Z'].min())

    # Apply weights to the z-scores, using (1 - Z) approach for drawdowns
    final_df['Final_Z_Score'] = (
        1 * final_df['Sortino_Ratio_Z'] + 
        1 * final_df['Dmonthly11_PnL_Z'] + 
        1 * (1 - final_df['Max_Drawdown_Z']) + 
        0.5 * final_df['Win%_Z']
    )

    # Sort by Final_Z_Score in descending order
    final_df = final_df.sort_values(by='Final_Z_Score', ascending=False)

    # Check for the top strategy with Sortino Ratio > 1
    top_strategy = None
    for _, row in final_df.iterrows():
        if row['Sortino Ratio'] > 1:
            top_strategy = row
            break

    if top_strategy is None:
        print("[WARNING] No strategy with Sortino Ratio > 1 found.")
    else:
        print(f"[INFO] Selected top strategy with Sortino Ratio > 1: {top_strategy['Filename']}")
        # Move top strategy to the top
        final_df = pd.concat([final_df.loc[[top_strategy.name]], final_df.drop(top_strategy.name)], ignore_index=True)

    return final_df


# Usage
# Assuming `existing_df` is the final DataFrame containing all selected strategies after appending each strategy
print(existing_df.columns)
final_sorted_df = compute_final_z_scores(existing_df)
analytics_folder = root_dir + f'/{stock}/{option_type}/Analytics'
os.makedirs(analytics_folder, exist_ok=True)
final_sorted_df.to_csv(root_dir + rf'/{stock}/{option_type}/Analytics/Analytics.csv', index=False)
