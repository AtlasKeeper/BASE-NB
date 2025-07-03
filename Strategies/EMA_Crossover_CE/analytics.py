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
stock = 'BANKNIFTY'
# stock = 'NIFTY'
# stock = 'FINNIFTY'                        
# option_type = 'CE'
option_type = 'CE'

superset = 'EMA_Crossover_CE'
# superset = 'EMA Crossover'
# superset = 'EMA MACD Crossover'
# superset = 'EMA MACD Support'
# superset = 'RSI'

# root_dir = f"/home/newberry/{superset} copy/"
root_dir = f"/home/newberry4/jay_test/{superset}/"
filter_df = pd.read_csv(root_dir + f'/{stock}/{option_type}/Filter_Sheets/filter_df3.csv')
# filter_df = pd.read_csv(root_dir + f'temp_data/{stock}/{option_type}/filter_df6.csv')
tradesheet_folder = root_dir + f'/{stock}/{option_type}/Trade_Sheets/'
output_folder = root_dir + f'/{stock}/{option_type}/final_tradesheet/'


####################################################################################################

os.makedirs(output_folder, exist_ok=True)

lot_size_dict = {'NIFTY': 25,'FINNIFTY': 25,
            'BANKNIFTY': 15}
govt_tc_dict = {"NIFTY": 2.25, 'FINNIFTY': 2.25 ,
           "BANKNIFTY": 3}

Date = {'start': '2021-06-01', 'end': '2024-05-10'}
total_months = 36


for index, row in filter_df.iterrows():
    strategy_file = row['Strategy']
    start_date = row['Start_Date']
    end_date = row['End_Date']

    # Check if the strategy file exists
    strategy_path = os.path.join(tradesheet_folder, strategy_file)
    if not os.path.exists(strategy_path + '.csv'):
        print(f"Strategy file {strategy_file}.csv not found.")
        continue

    try:
        strategy_data = pd.read_csv(strategy_path + '.csv')
    except:
        print('Data not available :', strategy_path)
        continue
    # Initialize an empty DataFrame for combined_filtered_data
    combined_filtered_data = pd.DataFrame()

    # Check which DTE parameter is set to 1 and filter the corresponding rows
    for i in range(5):
        dte_param = f'DTE{i}'
        if row[dte_param] == 1:
            filtered_data = strategy_data[strategy_data['WeeklyDaysToExpiry'] == i]
            combined_filtered_data = pd.concat([combined_filtered_data, filtered_data], ignore_index=True)

    # output file path
    output_file_path = os.path.join(output_folder, f"{strategy_file}.xlsx")

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exists. Skipping.")
        continue

    #saved in final_tradesheet
    combined_filtered_data.to_excel(output_file_path, index=False)


########################################### daily pnl ##################################


inputfolder_path = root_dir + f'/{stock}/{option_type}/final_tradesheet/'
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
          
    for key, range_tuple in range_dict.items():
        start, end = eval(key)
        if start <= abs(value) <= end:
            return abs(range_tuple * lot_size)
       
    return (abs(value * lot_size * 0.3) / 100)



for file in files:
    if file.endswith('.xlsx') or file.endswith('.xls'):
        file_path = os.path.join(inputfolder_path, file)
        #print(file_path)
        df = pd.read_excel(file_path)
        dfs[file] = df

for file, df in dfs.items():
    if file.startswith('NIFTY'):
        idx_calc = 'NIFTY'
    elif file.startswith("BANKNIFTY"):
        idx_calc = 'BANKNIFTY'
    elif file.startswith("FINNIFTY"):
        idx_calc = 'FINNIFTY'

    govt_charge = govt_tc_dict[idx_calc]
    lot_size = lot_size_dict[idx_calc]

    pnl_list = []
    for idx, row in df.iterrows():
        row_premium = row['Premium']
        row_tc = abs(get_spread_from_range(row_premium, idx_calc, lot_size)) + govt_charge
        row_pnl = (row_premium * lot_size) - (row_tc)
        pnl_list.append(row_pnl)

    df['PnL'] = pnl_list

    result_df = df.groupby(['Date', 'WeeklyDaysToExpiry']).agg({
        'Type': 'first',
        'ExpiryDate': 'first',
        'PnL': 'sum'
    }).reset_index()

    if not os.path.exists(outputfolder_path):
        os.makedirs(outputfolder_path)
    

    # Save the result_df as an Excel file in the 'dailypnl/' folder
    output_file_path = os.path.join(outputfolder_path, f'{file}')

    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exists. Skipping.")
        continue

    selected_columns = ['Date', 'WeeklyDaysToExpiry', 'Type', 'ExpiryDate', 'PnL']
    result_df[selected_columns].to_excel(output_file_path, index=False)


########################################################################################################################3
#filtering according to each period profitability

strategy_list = []
dte = [0, 1, 2, 3, 4]

#32 months
periods = [
    {'in_start': '2021-06-07', 'in_end': '2022-05-31', 'out_start': '2022-06-01', 'out_end': '2022-09-31'},
    {'in_start': '2021-10-01', 'in_end': '2022-09-31', 'out_start': '2022-10-01', 'out_end': '2023-01-31'},
    {'in_start': '2022-02-01', 'in_end': '2023-01-31', 'out_start': '2023-02-01 ', 'out_end': '2023-05-31'},
    {'in_start': '2022-06-01', 'in_end': '2023-05-31', 'out_start': '2023-06-01', 'out_end': '2023-09-31'},
    {'in_start': '2022-10-01', 'in_end': '2023-09-31', 'out_start': '2023-10-01 ', 'out_end': '2024-01-31'},
    {'in_start': '2023-02-01', 'in_end': '2024-01-31', 'out_start': '2024-02-01 ', 'out_end': '2024-05-10'}
]

#################################################################################################
# path to the folder
folder_path = root_dir + f'/{stock}/{option_type}/dailypnl/*.xlsx'
excel_filename = root_dir + f'{superset}_{stock}_{option_type}.xlsx'
#################################################################################################

file_paths = glob.glob(folder_path)

# for file_path in file_paths:
#     excel_data = pd.read_excel(file_path)
#     strategy_list.append(excel_data)

def read_file(filename):
    df = pd.read_excel(filename, engine='openpyxl')
    return df

# Create a dictionary to store the results
strategy_dict = {}

# Use multiprocessing to read files concurrently
with multiprocessing.Pool(processes=16) as pool:
    with tqdm(total=len(file_paths), desc='Processing', unit='Iteration') as pbar:
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            result = pool.apply_async(read_file, args=(file_path,))
            strategy_dict[filename] = result.get()
            pbar.update(1)


def calculate_period_pnl(df, dte_value, in_start, in_end, out_start, out_end):
    # Filter data based on dte_value
    filtered_data = df[df['WeeklyDaysToExpiry'] == dte_value]

    # Filter data based on date ranges
    in_period_data = filtered_data[(filtered_data['Date'] >= in_start) & (filtered_data['Date'] <= in_end)]
    out_period_data = filtered_data[(filtered_data['Date'] >= out_start) & (filtered_data['Date'] <= out_end)]

    # Convert 'PnL' column to numeric to handle non-numeric values
    in_period_data['PnL'] = pd.to_numeric(in_period_data['PnL'], errors='coerce')
    out_period_data['PnL'] = pd.to_numeric(out_period_data['PnL'], errors='coerce')

    # Calculate total PnL for each period
    in_period_pnl = in_period_data['PnL'].sum()
    out_period_pnl = out_period_data['PnL'].sum()

    return in_period_pnl, out_period_pnl

def walkforwardpnl():
    results = {dte_val: [] for dte_val in dte}  # Use a dictionary to store results for each dte

    for file_path, df in strategy_dict.items():
        unique_dte_values = df['WeeklyDaysToExpiry'].unique()

        for dte_value in unique_dte_values:
            for period in periods:
                in_start = period['in_start']
                in_end = period['in_end']
                out_start = period['out_start']
                out_end = period['out_end']

                in_pnl, out_pnl = calculate_period_pnl(df, dte_value, in_start, in_end, out_start, out_end)

                # if in_pnl <=0.0 or out_pnl <= 0.0:                      ## if want to filter for every part positive
                #     all_periods_positive = False
                #     break  
                # if all_periods_positive:
                # result = {
                #     'file':os.path.basename(file_path),
                #     'dte': dte_value,
                # }
                result = {
                    'file': os.path.basename(file_path),
                    'dte': dte_value,
                    'in_pnl': in_pnl,
                    'out_pnl': out_pnl
                }
                results[dte_value].append(result)  # Append result to the corresponding dte

    return results


result_dict = walkforwardpnl()
# Save results to Excel files, each dte in a separate sheet
with pd.ExcelWriter(excel_filename) as writer:
    for dte_value, result_list in result_dict.items():
        df = pd.DataFrame(result_list)
        df.to_excel(writer, sheet_name=f'dte_{dte_value}', index=False)





##################################### ANALYTICS CREATOR #########################################

# Date = {'start': '2021-06-07', 'end': '2023-09-30'}
# total_months = 28

def minPnl(Date, df, days_to_expiry):
    monthly31 = {'start': '2021-06-01', 'end': '2024-05-31'}
    monthly11 = {'start': '2023-06-01', 'end': '2024-05-31'}
    monthly3 = {'start': '2024-02-01', 'end': '2024-05-31'}

    def calculate_monthly_pnl(Date, df, days_to_expiry):
        start_date = Date['start']
        end_date = Date['end']

        mask = (df['WeeklyDaysToExpiry'] == days_to_expiry) & (df['Date'] >= start_date) & (df['Date'] <= end_date)
        monthly_pnls = df.loc[mask, 'PnL'].tolist()

        return sum(monthly_pnls)  # Sum the PnL values instead of returning the list

    # Pass the 'days_to_expiry' argument when calling calculate_monthly_pnl
    Dmonthly31_pnl = calculate_monthly_pnl(monthly31, df, days_to_expiry) / 36
    Dmonthly11_pnl = calculate_monthly_pnl(monthly11, df, days_to_expiry) / 12
    Dmonthly3_pnl = calculate_monthly_pnl(monthly3, df, days_to_expiry) / 4
    
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




def analytics(sheet_name, sub_dataframe_dailypnl ,sub_dataframe_tradesheet , output_folder, file_name):
    sub_dataframe_dailypnl['PnL'] = pd.to_numeric(sub_dataframe_dailypnl['PnL'], errors='coerce')

    if sheet_name.startswith('dte_'):
        days_to_expiry = int(sheet_name.split('_')[1])  # Extract the numeric part from sheet_name
        
        # Convert 'Date' column to datetime format if not already in datetime
        sub_dataframe_dailypnl['Date'] = pd.to_datetime(sub_dataframe_dailypnl['Date'], errors='coerce')
        
        totalpnl = sub_dataframe_dailypnl[sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry]['PnL'].sum()

        # Call get_drawdown function
        max_drawdown, max_drawdown_percentage, max_drawdown_date, time_to_recover, peak_date_before_max_drawdown = get_drawdown(
            sub_dataframe_dailypnl[sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry]['Date'],
            sub_dataframe_dailypnl[sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry]['PnL']
        )

        overall_min_pnl,Dmonthly31_pnl,Dmonthly11_pnl,Dmonthly3_pnl = minPnl(sub_dataframe_dailypnl['Date'],sub_dataframe_dailypnl, days_to_expiry)

        # Additional operations
        Profits = sub_dataframe_dailypnl[(sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry) & (sub_dataframe_dailypnl['PnL'] > 0)]['PnL']
        Losses = sub_dataframe_dailypnl[(sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry) & (sub_dataframe_dailypnl['PnL'] <= 0)]['PnL']

        total_trades = len(sub_dataframe_dailypnl[sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry])
        num_winners = len(Profits)
        num_losers = len(Losses)
        win_percentage = 100 * num_winners / total_trades
        loss_percentage = 100 * num_losers / total_trades

        max_profit = Profits.max() if num_winners > 0 else 0
        max_loss = Losses.min() if num_losers > 0 else 0

        median_pnl = sub_dataframe_dailypnl[sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry]['PnL'].median()
        median_profit = Profits.median() if num_winners > 0 else 0
        median_loss = Losses.median() if num_losers > 0 else 0

        sd_pnl = sub_dataframe_dailypnl[sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry]['PnL'].std()
        sd_profit = Profits.std() if num_winners > 0 else 0
        sd_loss = Losses.std() if num_losers > 0 else 0

        # Monthly PnL calculation
        # monthly_pnl = totalpnl / total_months
        # max investment Directional
        max_investment = -((sub_dataframe_tradesheet[sub_dataframe_tradesheet['WeeklyDaysToExpiry'] == days_to_expiry].loc[sub_dataframe_tradesheet['Action'] == 'long', 'Premium'].min()) * lot_size)
        
        roi_with_dd=100*overall_min_pnl*total_months/(max_investment + max_drawdown)   ####changed for total period roi calc
        roi = 100*overall_min_pnl*total_months/max_investment####changed for total period roi calc
        action = sub_dataframe_dailypnl[sub_dataframe_dailypnl['WeeklyDaysToExpiry'] == days_to_expiry]['Type'].unique()
      

        # Create a DataFrame with file, Total PnL, Max Drawdown, and additional metrics
        result_df = pd.DataFrame({
            'file': [os.path.basename(file_name)],
            'DTE':[days_to_expiry],
            'Action':action,
            'Total PnL': [totalpnl],
            'Max Drawdown': [max_drawdown],
            'Max Drawdown Percentage': [max_drawdown_percentage],
            '28M Monthly PnL' : [Dmonthly31_pnl],
            '12M Monthly PnL' : [Dmonthly11_pnl],
            'Daily 4M PnL ' : [Dmonthly3_pnl],
            'min pnl ' : [overall_min_pnl],
            'Max Investment': [max_investment],
            'ROI % ' : [roi],
            'ROI with DD' : [roi_with_dd],
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
            'SD Loss': [sd_loss]
        })

        # Save the DataFrame to an Excel file for each 'dte' sheet
        output_file_path = os.path.join(output_folder, f"{sheet_name}.xlsx")

        # If the file already exists, append the new result to it
        if os.path.isfile(output_file_path):
            existing_df = pd.read_excel(output_file_path)
            result_df = pd.concat([existing_df, result_df], ignore_index=True)

        result_df = result_df.drop_duplicates().reset_index().drop(columns = 'index')
        result_df.to_excel(output_file_path, index=False)

##################################################################################################
#INPUT FOLDER
dailypnl_folder = root_dir + f'/{stock}/{option_type}/dailypnl/'
tradesheet_folder = root_dir + f'/{stock}/{option_type}/final_tradesheet/'
analytics_folder = root_dir + f'/{stock}/{option_type}/Analytics/'

# All Daily PnL files
results_folder = f"/home/newberry4/jay_test/{superset}/result2/"

#OUTPUT FOLDER
os.makedirs(analytics_folder, exist_ok=True)

excel_file_path = excel_filename
##################################################################################################

workbook = pd.ExcelFile(excel_file_path)

# Iterate through each sheet
for sheet_name in workbook.sheet_names:
    sheet = workbook.parse(sheet_name)
    if sheet.empty:
        print(f"The sheet '{sheet_name}' is empty.")
    else:
        file_paths = sheet.iloc[:, 0]

        for file_name in file_paths:
            if 'stoploss_Band' in file_name:
                print(file_name)
                continue
            full_file_path_dailypnl = os.path.join(dailypnl_folder, file_name)
            full_file_path_tradesheet = os.path.join(tradesheet_folder, file_name)

            if os.path.isfile(full_file_path_dailypnl):
                sub_dataframe_dailypnl = pd.read_excel(full_file_path_dailypnl)
                # print(sub_dataframe_dailypnl)
            if os.path.isfile(full_file_path_tradesheet):
                sub_dataframe_tradesheet = pd.read_excel(full_file_path_tradesheet)
                # print(sub_dataframe_tradesheet)

                analytics(sheet_name, sub_dataframe_dailypnl,sub_dataframe_tradesheet, analytics_folder, file_name)
                output_file_path = os.path.join(analytics_folder, f"{sheet_name}.xlsx")
                result_df = pd.read_excel(output_file_path)
                result_df.sort_values(by='ROI with DD', ascending=False, inplace=True)
                # result_df = result_df.loc[result_df['Time to Recover'] <= 360]
                result_df.to_excel(output_file_path, index=False)
                result_df.to_excel(output_file_path, index=False)      

def read_excel_file(file_path):
    return pd.read_excel(file_path)

def calculate_ratio(pnl_value, max_investment):
    if max_investment != 0:
        return (pnl_value / max_investment) * 100
    else:
        return 0

def calculate_and_store_correlations(output_dataframe, dailypnl_folder):
    print("\n[DEBUG] Starting correlation calculation")

    result_dict = {}

    # Loop through each row in the output DataFrame
    for index, row in output_dataframe.iterrows():
        days_to_expiry = row['DTE']
        file_name_in_dailypnl = row['file']
        dailypnl_file_path = os.path.join(dailypnl_folder, file_name_in_dailypnl)

        print(f"[DEBUG] Processing file: {file_name_in_dailypnl} for DTE: {days_to_expiry}")

        try:
            # Read the file and filter by days_to_expiry
            dailypnl_dataframe = read_excel_file(dailypnl_file_path)
            dailypnl_dataframe = dailypnl_dataframe[dailypnl_dataframe['WeeklyDaysToExpiry'] == days_to_expiry]

            if dailypnl_dataframe.empty:
                print(f"[DEBUG] Warning: Data for {file_name_in_dailypnl} is empty after filtering for DTE {days_to_expiry}")
                continue

            pnl_value = dailypnl_dataframe.loc[:, 'PnL']
            max_investment = row['Max Investment']

            # Calculate the ratio
            ratio = calculate_ratio(pnl_value, max_investment)
            dailypnl_dataframe['PnL %'] = ratio

            # Store the result in the dictionary
            result_dict[file_name_in_dailypnl] = dailypnl_dataframe[['Date', 'PnL %']]

        except Exception as e:
            print(f"[DEBUG] Error processing {file_name_in_dailypnl}: {e}")
            continue

    print(f"[DEBUG] Completed result_dict: {list(result_dict.keys())}")

    # Calculate correlations
    correlations = {}

    if not result_dict:
        print("[DEBUG] result_dict is empty, skipping correlation calculation.")
        return correlations, None

    first_key = list(result_dict.keys())[0]
    first_list = result_dict[first_key]
    first_list['PnL %_first'] = first_list['PnL %']

    print(f"[DEBUG] First key for correlation: {first_key}")

    for other_key in result_dict.keys():
        if other_key != first_key:
            other_list = result_dict[other_key]
            other_list['PnL %_second'] = other_list['PnL %']

            print(f"[DEBUG] Calculating correlation between {first_key} and {other_key}")

            # Merge the data on 'Date'
            merged_df = pd.merge(first_list[['Date', 'PnL %_first']], other_list[['Date', 'PnL %_second']], on='Date', how='outer')

# Fill NaN values with 0 for missing PnL %
            merged_df['PnL %_first'].fillna(0, inplace=True)
            merged_df['PnL %_second'].fillna(0, inplace=True)

# Sort by 'Date' after the merge
            merged_df.sort_values(by='Date', inplace=True)

            if merged_df.empty:
                print(f"[DEBUG] No overlapping dates between {first_key} and {other_key}")
                continue

            merged_df = merged_df[~((merged_df['PnL %_first'] == 0) & (merged_df['PnL %_second'] == 0))]

            # Calculate correlation
            correlation = merged_df['PnL %_first'].corr(merged_df['PnL %_second'])

            print(f"[DEBUG] Correlation between {first_key} and {other_key}: {correlation}")

            if correlation is not None and -0.5 <= correlation <= 0.5:
                correlations[(first_key, other_key)] = correlation
                print(f"[DEBUG] Storing correlation between {first_key} and {other_key}")

    print(f"[DEBUG] Completed correlation calculations: {correlations}")
    return correlations, first_key


def process_and_calculate_correlations(output_folder, dailypnl_folder):
    print("\n[DEBUG] Starting to process files in output_folder")

    all_results_list = []

    for filename in os.listdir(output_folder):
        if filename.endswith(".xlsx"):
            print(f"[DEBUG] Processing file: {filename}")

            try:
                output_file_path = os.path.join(output_folder, filename)
                output_dataframe = read_excel_file(output_file_path)

                if output_dataframe.empty:
                    print(f"[DEBUG] Warning: {filename} is empty.")
                    continue

                # Drop duplicates and reset index
                output_dataframe = output_dataframe.drop_duplicates().reset_index().drop(columns='index')
                print(f"[DEBUG] {filename} loaded successfully with shape: {output_dataframe.shape}")

                # Calculate and store correlations
                correlations, first_key = calculate_and_store_correlations(output_dataframe, dailypnl_folder)

                if correlations:
                    # Append the dictionary of results to the list
                    all_results_list.append((filename, correlations, first_key))
                    print(f"[DEBUG] Correlations stored for {filename}")
                else:
                    print(f"[DEBUG] No correlations found for {filename}")

            except Exception as e:
                print(f"[DEBUG] Error processing {filename}: {e}")

    print("[DEBUG] Completed processing all files")
    return all_results_list

def selected_strategy_info(unique_keys_df, analytics_folder, dailypnl_folder, output_folder):
    info_list = []

# CHANGE selected_dailypnl_folder NAME ######################
    selected_dailypnl_folder = os.path.join(output_folder, f'{superset}_{stock}_{option_type}_dailypnl')
    os.makedirs(selected_dailypnl_folder, exist_ok=True)

    # Iterate over all files in the 'Analytics' folder
    for filename in os.listdir(analytics_folder):
        if filename.endswith(".xlsx"):
            
            analytics_file_path = os.path.join(analytics_folder, filename)
            analytics_dataframe = read_excel_file(analytics_file_path)

            analytics_dataframe['file_and_DTE'] = analytics_dataframe['file'] + '_' + analytics_dataframe['DTE'].astype(str)
            unique_keys_df['Keys_and_DTE'] = unique_keys_df['Keys'] + '_' + unique_keys_df['DTE'].astype(str)
            
            if 'file' in analytics_dataframe.columns:
                relevant_rows = unique_keys_df[unique_keys_df['Keys_and_DTE'].isin(analytics_dataframe['file_and_DTE'])]

                # Extract relevant information and append to info_list
                for index, row in relevant_rows.iterrows():
                    action_value = option_type
                    superset_value = superset

                    info = {
                        'Strategy': row['Keys'],
                        'Action': action_value,
                        'Max Investment': analytics_dataframe.loc[analytics_dataframe['file'] == row['Keys'], 'Max Investment'].iloc[0],
                        'WeeklyDaysToExpiry': analytics_dataframe.loc[analytics_dataframe['file'] == row['Keys'], 'DTE'].iloc[0],
                        'Superset': superset_value,
                    }
                    info_list.append(info)

                    dailypnl_file_name = row['Keys']
                    dailypnl_source_path = os.path.join(dailypnl_folder, dailypnl_file_name)
                    if os.path.exists(selected_dailypnl_folder):
                        dailypnl_dest_path = os.path.join(selected_dailypnl_folder, dailypnl_file_name)
                        shutil.copyfile(dailypnl_source_path, dailypnl_dest_path)
                    else:
                        os.makedirs(selected_dailypnl_folder, exist_ok=True)
                        shutil.copyfile(dailypnl_source_path, dailypnl_dest_path)
                        
    strategy_info_df = pd.DataFrame(info_list)


    # Check if the Excel file exists
    excel_file_path = os.path.join(output_folder, f'{stock}_{option_type}_strategy_info.xlsx')
    if os.path.exists(excel_file_path):
        # Load existing Excel file
        existing_df = pd.read_excel(excel_file_path)
        updated_df = pd.concat([existing_df, strategy_info_df], ignore_index=True)
        updated_df = updated_df.drop_duplicates().reset_index().drop(columns = 'index')
        updated_df.to_excel(excel_file_path, index=False)
        print("Data appended to existing Excel file.")
    else:
        strategy_info_df = strategy_info_df.drop_duplicates().reset_index().drop(columns = 'index')
        strategy_info_df.to_excel(excel_file_path, index=False)
        print("New Excel file created with the appended data.")




def main():

    print("Starting main() function")

    all_results_list = process_and_calculate_correlations(analytics_folder, dailypnl_folder)
    print("All results list loaded successfully:", all_results_list)

    unique_keys_df = pd.DataFrame(columns=['Keys', 'DTE'])
    counter_check = 1
    for filename, correlations, first_key in all_results_list:
        counter = 0  # Counter to track the number of correlations printed
        unique_keys = []
        dte_list = []
        DTE = int(filename.split('.xlsx')[0][-1])
        
        print("\nFilename Check:", filename)
        print("Counter Check:", counter_check)
        counter_check += 1

        print("First Key:", first_key)

        if not correlations:
            print("No correlations found for this file. Adding first_key to unique_keys.")
            unique_keys.append(first_key)
            dte_list.append(DTE)
            new_data = pd.DataFrame({'Keys': unique_keys, 'DTE': dte_list})
            unique_keys_df = pd.concat([unique_keys_df, new_data], ignore_index=True)
            continue

        for keys, correlation in correlations.items():
            print(f"Processing keys: {keys}, correlation: {correlation}")
            
            if counter < 1:
                print("Counter < 1, adding keys to unique_keys")

                # Add both keys to the list
                unique_keys.append(keys[0])
                dte_list.append(DTE)
                unique_keys.append(keys[1])
                dte_list.append(DTE)

                print(f"Filename: {filename}")
                print(f"  First Key: {keys[0]}, Other Key: {keys[1]}, Correlation: {correlation}")
                counter += 1

            elif counter == 1:
                print("Counter == 1, processing second and third sets")

                second_set = unique_keys[1]
                third_set = keys[1]

                print(f"Second Set: {second_set}, Third Set: {third_set}")

                try:
                    second_pnl = pd.read_excel(dailypnl_folder + '/' + second_set)
                    second_pnl = second_pnl[second_pnl['WeeklyDaysToExpiry'] == DTE]

                    third_pnl = pd.read_excel(dailypnl_folder + "/" + third_set)
                    third_pnl = third_pnl[third_pnl['WeeklyDaysToExpiry'] == DTE]

                    analytics_dataframe = pd.read_excel(analytics_folder + '/' + filename)
                    second_max = analytics_dataframe[analytics_dataframe['file'] == second_set]['Max Investment'].iloc[0]
                    third_max = analytics_dataframe[analytics_dataframe['file'] == third_set]['Max Investment'].iloc[0]

                    print(f"Second Max: {second_max}, Third Max: {third_max}")

                    second_pnl['PnL %_2'] = (second_pnl['PnL'] / second_max) * 100
                    third_pnl['PnL %_3'] = (third_pnl['PnL'] / third_max) * 100

                    merged_df = pd.merge(second_pnl[['Date', 'PnL %_2']], third_pnl[['Date', 'PnL %_3']], on='Date', how='outer')

                    merged_df['PnL %_2'].fillna(0, inplace=True)
                    merged_df['PnL %_3'].fillna(0, inplace=True)

                    merged_df = merged_df[~((merged_df["PnL %_2"] == 0) & (merged_df["PnL %_3"] == 0))]

                    second_third_correlation = merged_df['PnL %_2'].corr(merged_df['PnL %_3'])
                    print(f"Correlation between second and third sets: {second_third_correlation}")

                    if -0.5 < second_third_correlation < 0.5:
                        print("Correlation is in the range [-0.5, 0.5]. Adding third key to unique_keys.")
                        unique_keys.append(keys[1])
                        dte_list.append(DTE)

                        print(f"Filename: {filename}")
                        print(f"  First Key: {keys[0]}, Other Key: {keys[1]}, Correlation: {correlation}")
                        counter += 1
                        break

                except Exception as e:
                    print(f"Error processing second and third sets: {e}")

        new_data = pd.DataFrame({'Keys': unique_keys, 'DTE': dte_list})
        unique_keys_df = pd.concat([unique_keys_df, new_data], ignore_index=True)
        
        print("Updated unique_keys_df:")
        print(unique_keys_df)

    print("Final unique_keys_df:")
    print(unique_keys_df)

    selected_strategy_info(unique_keys_df, analytics_folder, dailypnl_folder, results_folder)
    print("Completed selected_strategy_info")

if __name__ == "__main__":
    main()
