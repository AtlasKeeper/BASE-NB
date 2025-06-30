# import datetime as dt
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
import talib as ta
import time
from tqdm import tqdm
from functools import partial
import os
from datetime import datetime, timedelta
import ast, json, sys, re
sys.path.insert(0, r"/home/newberry4/jay_data/")
from Common_Functions.utils import TSL, postgresql_query, resample_data, nearest_multiple, round_to_next_5_minutes
from Common_Functions.utils import get_target_stoploss, get_open_range, check_crossover, compare_month_and_year
import warnings
warnings.filterwarnings("ignore")

def postgresql_query(input_query, input_tuples = None):
    try:
        connection = psycopg2.connect(
            host="192.168.18.18",
            port = 5432,
            database="postgres",
            user="postgres",
            password="New@123",
        )
        
        cursor = connection.cursor()
        
        if input_tuples is not None:
            cursor.execute(input_query, input_tuples)
        else:
            cursor.execute(input_query)
        
        data = cursor.fetchall()
    
    except psycopg2.Error as e:
        print('Error connecting to the database:', e)
        return e
    
    else:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        
        return data

def resample_data(data, TIME_FRAME):
    
    resampled_data = data.resample(TIME_FRAME).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data

def nearest_multiple(x, n):
    remainder = x%n
    if remainder < n/2:
        nearest = x - remainder
    else:
        nearest = x - remainder + n
    return int(nearest)

def get_strike(ATM, minute, daily_option_data):
    
    def get_premium(option_type):
        
        subset = temp_option_data[(temp_option_data['StrikePrice'] == ATM) & (temp_option_data['Type'] == option_type)]
        
        if subset.empty:
            return None
        
        return subset['Open'].iloc[0]
    
    def find_nearest_strike(target_premium, option_type):
        
        subset = temp_option_data[temp_option_data['Type'] == option_type]
        
        subset = subset.reset_index(drop=True)
        if subset.empty:
            return None, None
        
        nearest_index = (subset['Open'] - target_premium).abs().idxmin()
        nearest_strike = subset.loc[nearest_index, 'StrikePrice']
        nearest_premium = subset.loc[nearest_index, 'Open']
        
        return nearest_strike, nearest_premium
    
    time = minute.strftime('%H:%M:%S')

    # Convert the filter time to a time object
    filter_time_object = pd.to_datetime(time).time()

    # Filter the DataFrame for the specific time
    temp_option_data = daily_option_data[daily_option_data.index.time == filter_time_object]
    #temp_option_data = daily_option_data[daily_option_data['Time'] == time]
    
    CE_ATM_premium = get_premium('CE')
    PE_ATM_premium = get_premium('PE')
    
    if (CE_ATM_premium is None) or (PE_ATM_premium is None):
        return None, None, None, None, None, None
    
    CE_OTM_premium_expected = CE_ATM_premium * STRIKE
    PE_OTM_premium_expected = PE_ATM_premium * STRIKE
    
    
    CE_OTM, CE_OTM_premium = find_nearest_strike(CE_OTM_premium_expected, 'CE')
    PE_OTM, PE_OTM_premium = find_nearest_strike(PE_OTM_premium_expected, 'PE')
    
    return CE_ATM_premium, PE_ATM_premium, CE_OTM, CE_OTM_premium, PE_OTM, PE_OTM_premium

# Function to get premium
def get_final_premium(premium_data, option_data, RATIO):
    option_data_ce = option_data[option_data['Type'] == 'CE'].drop(columns=['Ticker', 'Type'])
    option_data_pe = option_data[option_data['Type'] == 'PE'].drop(columns=['Ticker', 'Type'])
    
    del option_data

    premium_data['Time'] = pd.to_datetime(premium_data['Time'], format='%H:%M:%S')
    premium_data['Time'] = premium_data['Time'].dt.strftime('%H:%M')
    premium_data['ATM'] = premium_data['ATM'].astype('int32')

    premium_data = premium_data.merge(option_data_ce, left_on=['Date', 'Time', 'ExpiryDate', 'ATM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    
    premium_data = premium_data.rename(columns={'Open' : 'CE_ATM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_pe, left_on=['Date', 'Time', 'ExpiryDate', 'ATM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'PE_ATM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_ce, left_on=['Date', 'Time', 'ExpiryDate', 'CE_OTM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'CE_OTM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_pe, left_on=['Date', 'Time', 'ExpiryDate', 'PE_OTM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'PE_OTM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    long_lots = RATIO[0]
    short_lots = RATIO[1]

    premium_data['CE_ATM_Price'] = np.where(premium_data['Position'] == 1, long_lots * (-LOT_SIZE * premium_data['CE_ATM_Premium'] * 1.01 - brokerage), long_lots * (LOT_SIZE * premium_data['CE_ATM_Premium'] * 0.99 - brokerage))
    premium_data['PE_ATM_Price'] = np.where(premium_data['Position'] == 1, long_lots * (-LOT_SIZE * premium_data['PE_ATM_Premium'] * 1.01 - brokerage), long_lots * (LOT_SIZE * premium_data['PE_ATM_Premium'] * 0.99 - brokerage))
    premium_data['CE_OTM_Price'] = np.where(premium_data['Position'] == 1, short_lots * (LOT_SIZE * premium_data['CE_OTM_Premium'] * 0.99 - brokerage), short_lots * (-LOT_SIZE * premium_data['CE_OTM_Premium'] * 1.01 - brokerage))
    premium_data['PE_OTM_Price'] = np.where(premium_data['Position'] == 1, short_lots * (LOT_SIZE * premium_data['PE_OTM_Premium'] * 0.99 - brokerage), short_lots * (-LOT_SIZE * premium_data['PE_OTM_Premium'] * 1.01 - brokerage))

    premium_data['Premium'] = premium_data['CE_ATM_Price'] + premium_data['PE_ATM_Price'] + premium_data['CE_OTM_Price']  + premium_data['PE_OTM_Price']

    premium_data['Date'] = pd.to_datetime(premium_data['Date'], format='%Y-%m-%d')
    premium_data['ExpiryDate'] = pd.to_datetime(premium_data['ExpiryDate'], format='%Y-%m-%d')

    premium_data['DaysToExpiry'] = (premium_data['ExpiryDate'] - premium_data['Date']).dt.days
    premium_data['DaysToExpiry'] = np.where(premium_data['DaysToExpiry']==6, 4, np.where(premium_data['DaysToExpiry']==5, 3, premium_data['DaysToExpiry']))


    return premium_data

# Function to pull options data for specified date range 
def pull_options_data_d(start_date, end_date, option_data_path, stock):
            
    start_time = time.time()
    option_data_files = next(os.walk(option_data_path))[2]
    option_data = pd.DataFrame()

    for file in option_data_files:

        file1 = compare_month_and_year(start_date, end_date, file, stock)
              
        if not file1:
            continue

        temp_data = pd.read_pickle(option_data_path + file)[['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open','High' , 'Low' ,'Close', 'Ticker']]
        temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
        temp_data = temp_data.rename_axis('DateTime')
        option_data = pd.concat([option_data, temp_data])

    print('Option data columns :', option_data.columns)
    option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
    option_data['Type'] = option_data['Type'].astype('category')
    
    end_time = time.time()
    print('Time taken to pull Options data :', (end_time-start_time))

    return option_data

# Function to pull index data for specified date range
def pull_index_data(start_date_idx, end_date_idx, stock):

    start_time = time.time()
    print(start_date_idx, end_date_idx)
    table_name = stock + '_IDX'
    data = postgresql_query(f'''
                            SELECT "Open", "High", "Low", "Close", "Ticker"
                            FROM "{table_name}"
                            WHERE "Date" >= '{start_date_idx}'
                            AND "Date" <= '{end_date_idx}'
                            AND "Time" BETWEEN '09:15' AND '15:29'
                            ''')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time taken to get Index Data:', elapsed_time)

    column_names = ['Open', 'High', 'Low', 'Close', 'Ticker']
    index_data = pd.DataFrame(data, columns = column_names)
    index_data['Date'] = pd.to_datetime(index_data['Ticker'].str[0:8], format = '%Y%m%d').astype(str)

    df = index_data.merge(mapped_days, on = 'Date')
    df.index = pd.to_datetime(df['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
    df = df.rename_axis('DateTime')
    df = df.sort_index()
    df = df.drop_duplicates()
    
    return df

def trade_sheet_creator(mapped_days, option_data, df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, STRIKE, ENTRY, EXIT):
    
    column_names = ['Date', 'Position', 'Action', 'CE_Time', 'PE_Time', 'Index Value', 'CE_OTM', 'PE_OTM', 'Exit', 'ExpiryDate', 'DaysToExpiry', 'CE_OTM_Premium', 'PE_OTM_Premium', 'EXIT_TYPE']
    trade_sheet = []

    # Parameters for filtering combinations
    target_list = [TIME_FRAME, STRIKE, ENTRY, EXIT]

    # Filter mapped days
    mapped_days_temp = mapped_days[
    (mapped_days['Date'] >= start_date) & 
    (mapped_days['Date'] <= end_date) &
    (mapped_days['Date'] > '2021-06-03') & 
    (mapped_days['Date'] != '2024-05-18') & 
    (mapped_days['Date'] != '2024-05-20') &
    ~((mapped_days['Date'] >= '2024-05-31') & (mapped_days['Date'] <= '2024-06-06'))
    ]

    for _, row in mapped_days_temp.iterrows():
        date = row['Date']
        print(date)
        # expiry_date = row['MonthlyExpiry']
        expiry_date = row['ExpiryDate']
        days_to_expiry = row['DaysToExpiry']

        start_time = pd.to_datetime(f'{date} {ENTRY}:00')
        end_time = pd.to_datetime(f'{expiry_date} {EXIT}:00')

        daily_data_start_time = start_time - pd.Timedelta('5T')
        daily_data = df[(df.index >= daily_data_start_time) & (df.index <= end_time)]

        filter_start_date = pd.to_datetime(date)
        filter_end_date = pd.to_datetime(expiry_date)
        daily_option_data = option_data[(option_data.index.date >= filter_start_date.date()) & 
                                        (option_data.index.date <= filter_end_date.date())]

        position = 0
        entry_time = start_time
        reentry_count = 0  # Initialize re-entry counter

        for minute, dd_row in daily_data.iloc[1:-1].iterrows():
            current_index_open = dd_row['Open']
            if position == 1:
                break

            # Entry
            if (position == 0) & (minute == entry_time):
                position = 1

                if ((date >= '2022-10-19') & (stock == 'FINNIFTY')) or (stock == 'NIFTY'):
                    ATM = nearest_multiple(current_index_open, 50)
                    ATM = nearest_multiple(ATM, 50)
                    CE_OTM = ATM * (1 + 0.03)
                    CE_OTM = nearest_multiple(CE_OTM, 50)
                    PE_OTM = ATM * (1 - 0.03)
                    PE_OTM = nearest_multiple(PE_OTM, 50)
                else:
                    ATM = nearest_multiple(current_index_open, 100)
                    CE_OTM = ATM + (STRIKE * 2)
                    PE_OTM = ATM - (STRIKE * 2)

                CE_OTM_entry_price, CE_OTM_exit_price, ce_exit_time_period, PE_OTM_entry_price, PE_OTM_exit_price, pe_exit_time_period, exit_type = TSL_SS(daily_option_data, minute, CE_OTM, PE_OTM, EXIT,days_to_expiry)

                minute_time = minute.time()

                if isinstance(ce_exit_time_period, int) or ce_exit_time_period == 0:
                    ce_exit_time_period = end_time
                if isinstance(pe_exit_time_period, int) or pe_exit_time_period == 0:
                    pe_exit_time_period = end_time

                ce_exit_time = ce_exit_time_period.time()
                pe_exit_time = pe_exit_time_period.time()

                trade_sheet.append(pd.Series([date, 1, 'Short', minute_time, minute_time, current_index_open, CE_OTM, PE_OTM, '', expiry_date, days_to_expiry, CE_OTM_entry_price, PE_OTM_entry_price, ''], index=column_names))
                trade_sheet.append(pd.Series([date, 0, 'Long', ce_exit_time, pe_exit_time, current_index_open, CE_OTM, PE_OTM, '', expiry_date, days_to_expiry, CE_OTM_exit_price, PE_OTM_exit_price,exit_type], index=column_names))

                # RE-ENTRY
                # exit_time_check = (datetime.strptime(EXIT, "%H:%M") - timedelta(minutes=5)).time()

                # reentry_time_period = ce_exit_time_period if ce_exit_time > pe_exit_time else pe_exit_time_period
                # reentry_time_period = round_to_next_5_minutes(reentry_time_period, '5T')  ### Check this
                # reentry_time = reentry_time_period.time()

                # while (reentry_time < exit_time_check) and (reentry_count < RENTRY):
                #     reentry_count += 1

                #     current_index_open = daily_data.at[reentry_time_period, 'Open']

                #     if ((date >= '2022-10-19') & (stock == 'FINNIFTY')) or (stock == 'NIFTY'):
                #         ATM = nearest_multiple(current_index_open, 50)
                #         CE_OTM = ATM + STRIKE
                #         PE_OTM = ATM - STRIKE
                #     else:
                #         ATM = nearest_multiple(current_index_open, 100)
                #         CE_OTM = ATM + (STRIKE * 2)
                #         PE_OTM = ATM - (STRIKE * 2)

                #     CE_OTM_entry_price, CE_OTM_exit_price, ce_exit_time_period, PE_OTM_entry_price, PE_OTM_exit_price, pe_exit_time_period, ce_stoploss, pe_stoploss, exit_type = TSL_SS(daily_option_data, reentry_time_period, CE_OTM, PE_OTM, EXIT, STOPLOSS_PT,TARGET , days_to_expiry)

                #     if isinstance(ce_exit_time_period, int) or ce_exit_time_period == 0:
                #         ce_exit_time_period = end_time
                #     if isinstance(pe_exit_time_period, int) or pe_exit_time_period == 0:
                #         pe_exit_time_period = end_time

                #     ce_exit_time = ce_exit_time_period.time()
                #     pe_exit_time = pe_exit_time_period.time()

                #     trade_sheet.append(pd.Series([date, 1, 'Short', reentry_time, reentry_time, current_index_open, CE_OTM, PE_OTM, '', expiry_date, days_to_expiry, CE_OTM_entry_price, PE_OTM_entry_price, ce_stoploss, pe_stoploss,''], index=column_names))
                #     trade_sheet.append(pd.Series([date, 0, 'Long', ce_exit_time, pe_exit_time, current_index_open, CE_OTM, PE_OTM, '', expiry_date, days_to_expiry, CE_OTM_exit_price, PE_OTM_exit_price, ce_stoploss, pe_stoploss,exit_type], index=column_names))

                #     reentry_time_period = ce_exit_time_period if ce_exit_time > pe_exit_time else pe_exit_time_period
                #     reentry_time_period = round_to_next_5_minutes(reentry_time_period, PREMIUM_TP)
                #     reentry_time = reentry_time_period.time()

    strategy_name = f'{stock}_candle_{TIME_FRAME}_strike_{STRIKE}_entry_{ENTRY}_exit_{EXIT}'
    sanitized_strategy_name = strategy_name.replace('.', ',').replace(':', ',')

    
    try:
        trade_sheet = pd.concat(trade_sheet, axis = 1).T
    except Exception as e:
        print(f"An error occurred: {e}")
        return sanitized_strategy_name + '_' + start_date + '_' + end_date

    # trade_sheet = get_final_premium(trade_sheet, option_data, RATIO)
    # trade_sheet['Time'] = pd.to_datetime(trade_sheet['Time'], format='%H:%M')
    # trade_sheet['Time'] = trade_sheet['Time'].dt.strftime('%H:%M:%S')

    trade_sheet['Premium'] = np.where(trade_sheet['Action']=='Short', trade_sheet['CE_OTM_Premium'] + trade_sheet['PE_OTM_Premium'], - (trade_sheet['CE_OTM_Premium'] + trade_sheet['PE_OTM_Premium']))

    # create filter_df to store profitable combo and dte
    filter_df1 = pd.DataFrame(columns=['Strategy', 'Parameters', 'DTE0', 'DTE1', 'DTE2', 'DTE3', 'DTE4', 'Status'])
    filter_df1.loc[len(filter_df1), 'Strategy'] = sanitized_strategy_name
    row_index = filter_df1.index[filter_df1['Strategy'] == sanitized_strategy_name].tolist()[0]
    filter_df1.loc[row_index, 'Parameters'] = target_list
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 0
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Start_Date'] = start_date
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'End_Date'] = end_date
    
    trade_sheet = trade_sheet[trade_sheet['Date'] > '2021-06-03']

    # Go through each dte for the current combo to check if it's profitable
    # for dte in dte_list:
        
    #     trade_sheet_temp = trade_sheet[trade_sheet['DaysToExpiry'] == dte]
    if not trade_sheet.empty:
        trade_sheet.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
            # if trade_sheet_temp['Premium'].sum() > 0:
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 1
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 1
                
                # trade_sheet_temp.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
            # else:
            #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 0

    # Store the combo and it's dte which is profitable in filter_df file
    # if filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'].iloc[0] == 1:
        
    existing_csv_file = rf"{filter_df_path}/filter_df{counter}.csv"
    if os.path.isfile(existing_csv_file):
        filter_df1.to_csv(existing_csv_file, index=False, mode='a', header=False)
    else:
        filter_df1.to_csv(existing_csv_file, index=False)
        
    return sanitized_strategy_name + '_' + str(start_date) + '_' + str(end_date)



def TSL_SS(option_data, next_time_period, CE_OTM, PE_OTM, EXIT, days_to_expiry): 
    from copy import deepcopy
    import pandas as pd

    # start_time = next_time_period
    # end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10] + ' ' + EXIT + ':00')

    # # Filter option data for the date, time, strike, and type of the entry position
    # intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]

    intraday_data = option_data


    # Separate CE and PE data
    intraday_data_ce = intraday_data[(intraday_data['StrikePrice'] == CE_OTM) & (intraday_data['Type'] == 'CE')]
    intraday_data_ce = intraday_data_ce.sort_index()

    intraday_data_pe = intraday_data[(intraday_data['StrikePrice'] == PE_OTM) & (intraday_data['Type'] == 'PE')]
    intraday_data_pe = intraday_data_pe.sort_index()

    # Get entry prices
    if intraday_data_ce.empty:
        return [0, 0, 0, 0, 0, 0, "No CE Data"]  # Return 0s if CE data is missing
    else:
        CE_OTM_entry_price = intraday_data_ce.iloc[0]['Open']
        intraday_data_ce = intraday_data_ce[intraday_data_ce.index > next_time_period]

    if intraday_data_pe.empty:
        return [0, 0, 0, 0, 0, 0, "No PE Data"]  # Return 0s if PE data is missing
    else:
        PE_OTM_entry_price = intraday_data_pe.iloc[0]['Open']
        intraday_data_pe = intraday_data_pe[intraday_data_pe.index > next_time_period]

    # Combine CE and PE data
    df_trade_combined = deepcopy(pd.concat([intraday_data_ce, intraday_data_pe], axis=1, keys=['CE', 'PE']))

    # Initialize stop-loss levels (multiplicative factor)
    # ce_stoploss = CE_OTM_entry_price * (1 + STOPLOSS_PT)
    # pe_stoploss = PE_OTM_entry_price * (1 + STOPLOSS_PT)

    # # Calculate combined target if TARGET is not 'NA' and days_to_expiry > 0
    # combined_target = None
    # if TARGET != 'NA' and days_to_expiry > 0:
    #     combined_target = (CE_OTM_entry_price + PE_OTM_entry_price) * (1 - TARGET)

    # Calculate individual targets if days_to_expiry == 0
    # ce_target = CE_OTM_entry_price * (1 - TARGET) if TARGET != 'NA' else None
    # pe_target = PE_OTM_entry_price * (1 - TARGET) if TARGET != 'NA' else None

    # Iterate over the combined data to check for stop-loss or target hit
    # for index, row in df_trade_combined.iterrows():
    #     ce_open_price = row['CE']['Open']  # Check CE Open price
    #     pe_open_price = row['PE']['Open']  # Check PE Open price
    #     ce_high_price = row['CE']['High']  # Check CE High price
    #     pe_high_price = row['PE']['High']  # Check PE High price
    #     ce_close_price = row['CE']['Close']  # Check CE Close price
    #     pe_close_price = row['PE']['Close']  # Check PE Close price

        # # Check stop-loss condition
        # if ce_high_price >= ce_stoploss or pe_high_price >= pe_stoploss:
        #     # Exit both legs at stop-loss + 7% of stop-loss using close price
        #     CE_OTM_exit_price = ce_stoploss * 1.05 if ce_high_price >= ce_stoploss else ce_high_price
        #     PE_OTM_exit_price = pe_stoploss * 1.05 if pe_high_price >= pe_stoploss else pe_high_price
        #     exit_time = index

        #     return [
        #         CE_OTM_entry_price, CE_OTM_exit_price, exit_time,
        #         PE_OTM_entry_price, PE_OTM_exit_price, exit_time,
        #         ce_stoploss, pe_stoploss,
        #         "Stop-Loss Hit"
        #     ]

        # # Check individual target condition if days_to_expiry == 0
        # if days_to_expiry == 0 and TARGET != 'NA':
        #     if ce_open_price <= ce_target or pe_open_price <= pe_target:
        #         CE_OTM_exit_price = ce_open_price
        #         PE_OTM_exit_price = pe_open_price
        #         exit_time = index

        #         return [
        #             CE_OTM_entry_price, CE_OTM_exit_price, exit_time,
        #             PE_OTM_entry_price, PE_OTM_exit_price, exit_time,
        #             ce_stoploss, pe_stoploss,
        #             "Individual Target Hit"
        #         ]

        # # Check combined target condition if days_to_expiry > 0
        # if combined_target is not None and TARGET != 'NA':
        #     combined_current = ce_open_price + pe_open_price  # Use open price for combined target
        #     if combined_current <= combined_target:
        #         CE_OTM_exit_price = ce_open_price
        #         PE_OTM_exit_price = pe_open_price
        #         exit_time = index

        #         return [
        #             CE_OTM_entry_price, CE_OTM_exit_price, exit_time,
        #             PE_OTM_entry_price, PE_OTM_exit_price, exit_time,
        #             ce_stoploss, pe_stoploss,
        #             "Combined Target Hit"
        #         ]

    # If no stop-loss or target hit, exit at the end time
    try:
        CE_OTM_exit_price = intraday_data_ce.iloc[-1]['Close']
        PE_OTM_exit_price = intraday_data_pe.iloc[-1]['Close']
        exit_time = intraday_data_ce.iloc[-1].name
    except IndexError:
        return [0, 0, 0, 0, 0, 0, "No Exit Data"]  # Return 0s if exit data is missing

    return [
        CE_OTM_entry_price, CE_OTM_exit_price, exit_time,
        PE_OTM_entry_price, PE_OTM_exit_price, exit_time,
        "Time Exit"
    ]




def TSL_SS2(option_data, next_time_period, CE_OTM, PE_OTM, EXIT, STOPLOSS_PT, PREMIUM_TP):
    
    start_time = next_time_period
    end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')

    # Filter option data for the date, time, strike and type of the entry position    
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]

    intraday_data_ce = intraday_data[(intraday_data['StrikePrice']==CE_OTM) & (intraday_data['Type']=='CE')]    
    intraday_data_ce = intraday_data_ce.sort_index()

    intraday_data_pe = intraday_data[(intraday_data['StrikePrice']==PE_OTM) & (intraday_data['Type']=='PE')]    
    intraday_data_pe = intraday_data_pe.sort_index()

    try:
        CE_OTM_entry_price = intraday_data_ce.iloc[0]['Open']
        intraday_data_ce = intraday_data_ce[intraday_data_ce.index > next_time_period]
    except:
        print(start_time, CE_OTM)
        print(intraday_data_ce)
        CE_OTM_entry_price = intraday_data_ce.iloc[0]['Open']
        intraday_data_ce = intraday_data_ce[intraday_data_ce.index > next_time_period]

    PE_OTM_entry_price = intraday_data_pe.iloc[0]['Open']
    intraday_data_pe = intraday_data_pe[intraday_data_pe.index > next_time_period]
    
    PE_OTM_new_entry_price = CE_OTM_new_entry_price = ''

    if PREMIUM_TP=='MIN':
        if CE_OTM_entry_price < PE_OTM_entry_price:
            ce_target_pt = CE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
            pe_target_pt = PE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
        
        else:
            ce_target_pt = CE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT
            pe_target_pt = PE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT
    else:
            ce_target_pt = CE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
            pe_target_pt = PE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT

    crosses_threshold_ce = (intraday_data_ce['Open'] > ce_target_pt)
    crosses_threshold_pe = (intraday_data_pe['Open'] > pe_target_pt)

    if crosses_threshold_ce.any() & crosses_threshold_pe.any() & (crosses_threshold_ce.idxmax() < crosses_threshold_pe.idxmax()):

        if intraday_data_ce[intraday_data_ce.index > crosses_threshold_ce.idxmax()].empty:
            ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
            except:    
                print(ce_exit_time_period)
                next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']
            ce_stoploss = 0
            
        else:
            crosses_threshold_ce_index = crosses_threshold_ce.idxmax()
            try:
                CE_OTM_exit_price = intraday_data_ce.at[crosses_threshold_ce_index + pd.to_timedelta('1T'), 'Open']
            except:
                print(crosses_threshold_ce_index + pd.to_timedelta('1T'))
                next_index = intraday_data_ce.index[intraday_data_ce.index < (crosses_threshold_ce_index + pd.to_timedelta('1T'))].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']

            ce_exit_time_period = crosses_threshold_ce_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            ce_stoploss = 1

            if crosses_threshold_ce.idxmax() < crosses_threshold_pe.idxmax():

                intraday_data_pe = intraday_data_pe.sort_index()
                intraday_data_pe = intraday_data_pe[intraday_data_pe.index >= ce_exit_time_period]

                PE_OTM_new_entry_price = intraday_data_pe.iloc[0]['Open'] 
                pe_target_pt = PE_OTM_new_entry_price * (1 + STOPLOSS_PT)

                crosses_threshold_pe = (intraday_data_pe['Open'] > pe_target_pt)

                if crosses_threshold_pe.any():
                    if intraday_data_pe[intraday_data_pe.index > crosses_threshold_pe.idxmax()].empty:
                        pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                        try:
                            PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
                        except:
                            next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
                        
                        pe_stoploss = 0
                    else:
                        crosses_threshold_pe_index = crosses_threshold_pe.idxmax()
                        try:
                            PE_OTM_exit_price = intraday_data_pe.at[crosses_threshold_pe_index + pd.to_timedelta('1T'), 'Open']
                        except:
                            next_index = intraday_data_pe.index[intraday_data_pe.index < (crosses_threshold_pe_index + pd.to_timedelta('1T'))].max()
                            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
                        
                        pe_exit_time_period = crosses_threshold_pe_index + pd.to_timedelta('1T')
                        exit_type = 'Initial Target Exit'
                        pe_stoploss = 1                
                else:
                    pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                    try:    
                        PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
                    except:
                        next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                        PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open'] 
                    
                    pe_stoploss = 0

    elif crosses_threshold_pe.any() & crosses_threshold_ce.any() & (crosses_threshold_pe.idxmax() < crosses_threshold_ce.idxmax()):
        if intraday_data_pe[intraday_data_pe.index > crosses_threshold_pe.idxmax()].empty:
            pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_stoploss = 0
        else:
            crosses_threshold_pe_index = crosses_threshold_pe.idxmax()
            try:
                PE_OTM_exit_price = intraday_data_pe.at[crosses_threshold_pe_index + pd.to_timedelta('1T'), 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < (crosses_threshold_pe_index + pd.to_timedelta('1T'))].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_exit_time_period = crosses_threshold_pe_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            pe_stoploss = 1

            if crosses_threshold_pe.idxmax() < crosses_threshold_ce.idxmax():

                intraday_data_ce = intraday_data_ce.sort_index()
                intraday_data_ce = intraday_data_ce[intraday_data_ce.index >= pe_exit_time_period]

                CE_OTM_new_entry_price = intraday_data_ce.iloc[0]['Open'] 
                ce_target_pt = CE_OTM_new_entry_price * (1 + STOPLOSS_PT)

                crosses_threshold_ce = (intraday_data_ce['Open'] > ce_target_pt)
                update_sl = 1

                if crosses_threshold_ce.any():
                    if intraday_data_ce[intraday_data_ce.index > crosses_threshold_ce.idxmax()].empty:
                        ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                        try:
                            CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
                        except:    
                            print(ce_exit_time_period)
                            next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']
                        ce_stoploss = 0
                    else:
                        crosses_threshold_ce_index = crosses_threshold_ce.idxmax()
                        try:
                            CE_OTM_exit_price = intraday_data_ce.at[crosses_threshold_ce_index + pd.to_timedelta('1T'), 'Open']
                        except:
                            print(crosses_threshold_ce_index + pd.to_timedelta('1T'))
                            next_index = intraday_data_ce.index[intraday_data_ce.index < (crosses_threshold_ce_index + pd.to_timedelta('1T'))].max()
                            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']

                        ce_exit_time_period = crosses_threshold_ce_index + pd.to_timedelta('1T')
                        exit_type = 'Initial Target Exit'
                        ce_stoploss = 1                   
                else:
                    ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10] + ' ' + EXIT + ':00')
                    try:
                        CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
                    except KeyError:
                        # print(start_time, end_time, CE_OTM, ce_exit_time_period)
                        next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                        CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open'] 

                    ce_stoploss = 0
    elif crosses_threshold_ce.any() & (not crosses_threshold_pe.any()):
        if intraday_data_ce[intraday_data_ce.index > crosses_threshold_ce.idxmax()].empty:
            ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
            except:    
                print(ce_exit_time_period)
                next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']
            ce_stoploss = 0
        else:
            crosses_threshold_ce_index = crosses_threshold_ce.idxmax()
            try:
                CE_OTM_exit_price = intraday_data_ce.at[crosses_threshold_ce_index + pd.to_timedelta('1T'), 'Open']
            except:
                print(crosses_threshold_ce_index + pd.to_timedelta('1T'))
                next_index = intraday_data_ce.index[intraday_data_ce.index < (crosses_threshold_ce_index + pd.to_timedelta('1T'))].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']

            ce_exit_time_period = crosses_threshold_ce_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            ce_stoploss = 1   

        pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:    
            PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
        except:
            next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open'] 
        
        pe_stoploss = 0
    
    elif crosses_threshold_pe.any() & (not crosses_threshold_ce.any()):
        if intraday_data_pe[intraday_data_pe.index > crosses_threshold_pe.idxmax()].empty:
            pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_stoploss = 0
        else:
            crosses_threshold_pe_index = crosses_threshold_pe.idxmax()
            try:
                PE_OTM_exit_price = intraday_data_pe.at[crosses_threshold_pe_index + pd.to_timedelta('1T'), 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < (crosses_threshold_pe_index + pd.to_timedelta('1T'))].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_exit_time_period = crosses_threshold_pe_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            pe_stoploss = 1

        ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:
            CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
        except KeyError:
            # print(start_time, end_time, CE_OTM, ce_exit_time_period)
            next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open'] 

        ce_stoploss = 0
    else:
        ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:
            CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
        except KeyError:
            next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open'] 
        ce_stoploss = 0

        pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:    
            PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
        except:
            next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open'] 
        pe_stoploss = 0

    del intraday_data_ce
    del intraday_data_pe
    return [CE_OTM_entry_price, CE_OTM_exit_price, ce_exit_time_period, PE_OTM_entry_price, PE_OTM_exit_price, pe_exit_time_period, ce_stoploss, pe_stoploss, CE_OTM_new_entry_price, PE_OTM_new_entry_price]



def resample_data(data, TIME_FRAME):
    
    resampled_data = data.resample(TIME_FRAME).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data

########################################### INPUTS #####################################################
# Inputs
superset = 'plain_vanilla'
stock = 'NIFTY'
option_type = 'ND'

roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' or stock == 'SENSEX'  else None)
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' or stock == 'SENSEX' else None)
LOT_SIZE = 25 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else 10)

# Define all the file paths
root_path = rf"/home/newberry4/jay_test/SHORT_STRADDLE_rajesh_sir/{superset}/{stock}/{option_type}/"
filter_df_path = rf"{root_path}/Filter_Sheets/"
# option_data_path = rf"/home/newberry2/Sourav/Data/{stock}/Current_Expiry//"

expiry_file_path = rf"/home/newberry4/jay_data/Common_Files/{stock} market dates.xlsx"
# expiry_file_path = rf"/home/newberry4/jay_data/updated_FINNIFTY_market_dates.xlsx"          # for finnifty
# expiry_file_path = "/home/newberry4/jay_data/BANKNIFTY market dates (1).xlsx"   # for banknifty
txt_file_path = rf'{root_path}/new_done.txt'
# output_folder_path = rf'{root_path}/Trade_Sheets/'
output_folder_path = rf'{root_path}/Trade_Sheets/'



if stock == 'NIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/Data/{stock}/Current_Expiry/"
    option_data_path = rf"/home/newberry4/jay_data/Data/NIFTY/NIFTY_OHLCV/NIFTY_OHLCV/"
elif stock =='BANKNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/BANKNIFTY_DATA/BANKNIFTY_OHLCV/"
    option_data_path = rf"/home/newberry4/jay_data/Data/BANKNIFTY/monthly_expiry/"
elif stock =='FINNIFTY':
    # option_data_path = rf"/home/newberry4/jay_data/FINNIFTY_2/"
    option_data_path = rf"/home/newberry4/jay_data/Data/FINNIFTY/monthly_expiry/"
elif stock =='SENSEX':
    option_data_path = rf"jay_data/Data/SENSEX/weekly_expiry/"

# Create all the required directories
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None

# start_date = '2022-06-01'
# end_date = '2022-09-30'

# list of period buckets
# date_ranges = [('2024-08-01', '2024-08-31')]


# date_ranges = [('2024-08-01', '2024-11-30'),('2024-04-01', '2024-07-31'),('2024-01-01', '2024-03-31'),('2023-09-13', '2023-12-31')]   # for sensex
#                ]   


# date_ranges = [('2023-09-13', '2023-09-30')]


date_ranges = [ ('2024-06-01', '2024-10-30'),
               ('2024-02-01', '2024-05-31'),
                ('2023-10-01', '2024-01-31'),
                ('2023-06-01', '2023-09-30'),
                ('2023-02-01', '2023-05-31'),
                ('2022-10-01', '2023-01-31'),
                ('2022-06-01', '2022-09-30'),
                ('2022-01-01', '2022-05-31'), 
                ('2021-06-01', '2021-12-31')]


# dte_list = [0]
# dte_list = [2, 3, 4]

# Final Combinations
candle_time_frame = ['5T']
entries = ['09:25']
exits = ['15:20']
# re_entries = ['1T','5T']
strikes = [0]            
# stoploss_per = [1.5 , 1.7 ,2]
# premium_type = ['5T']   #re_entries
# target = ['NA',0.2,0.4,0.6,0.8]
# rentry_limit = [2,3,4,5]


# Testing Combinations
# candle_time_frame = ['5T']
# entries = ['09:30']
# exits = ['15:15']
# re_entries = ['5T']
# strikes = [0]
# stoploss_per = [0.3]
# # profit_per = ['NA']
# premium_threshold = [10]
# premium_type = ['MIN']

parameters = []

def parameter_process(parameter, mapped_days, option_data, df, start_date, end_date, counter, output_folder_path):
    TIME_FRAME, STRIKE, ENTRY, EXIT = parameter
    resampled_df = resample_data(df,TIME_FRAME) 
    resampled = resampled_df.dropna()    
    return trade_sheet_creator(mapped_days, option_data, df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, STRIKE, ENTRY, EXIT )

if stock == 'BANKNIFTY' or stock == 'SENSEX':
    strikes = [x * 2 for x in strikes]

if __name__ == "__main__":
    
    counter = 0
    start_date_idx = date_ranges[-1][0]
    end_date_idx = date_ranges[0][-1]

    # Read expiry file
    mapped_days = pd.read_excel(expiry_file_path)
    mapped_days = mapped_days[(mapped_days['Date'] >= start_date_idx) & (mapped_days['Date'] <= end_date_idx)]
    mapped_days = mapped_days.rename(columns={'WeeklyDaysToExpiry' : 'DaysToExpiry'})

    # Pull Index Data
    df = pull_index_data(start_date_idx, end_date_idx, stock)
    resampled_df_main = resample_data(df, '5T')

    for start_date, end_date in date_ranges: 
         
        counter += 1
        print(start_date, end_date, counter)
 

       
        start_date_object = pd.to_datetime(start_date)
        end_date_object = pd.to_datetime(end_date)

        # Add 30 days to end_date_object
        new_end_date_object = end_date_object + timedelta(days=30)

        # Convert back to string in desired format
        new_end_date = new_end_date_object.strftime('%Y-%m-%d')

        # Pull options data
        # Ensure pull_options_data_d function expects strings in '%Y-%m-%d' format
        option_data = pull_options_data_d(start_date, new_end_date, option_data_path, stock)
        # option_data_ce = option_data[option_data['Type']=='CE']
        # option_data_pe = option_data[option_data['Type']=='PE']


        # print(option_data_ce)
        # print(option_data_pe)
        # Read/Create parameters to run
        
        parameters = []

        if counter==1:
            filter_df = pd.DataFrame()
        elif counter>1:
            if not os.path.exists(f"{filter_df_path}/filter_df{counter-1}.csv"):
                print(f"File filter_df{counter-1}.csv does not exist. Stopping the code.")
                sys.exit()
            else:
                filter_df = pd.read_csv(f"{filter_df_path}/filter_df{counter-1}.csv")  
                filter_df = filter_df.drop_duplicates()   
                    
        if counter!=1:
            parameters = filter_df['Parameters'].to_list()
            parameters = [ast.literal_eval(item.replace("'", "\"")) for item in parameters]
        
        
        elif counter==1:
            for TIME_FRAME in candle_time_frame:
                # for PREMIUM_TP in premium_type:
                for STRIKE in strikes:
                    for ENTRY in entries:
                        for EXIT in exits:
                            if ENTRY < EXIT:
                                    # for STOPLOSS_PT in stoploss_per:
                                    #     for TARGET in target:
                                    #         for RENTRY in rentry_limit:
                                parameters.append([TIME_FRAME,  STRIKE, ENTRY, EXIT])

        # Read the content of the log file to check which parameters have already been processed
        print('Total parameters :', len(parameters))
        file_path = txt_file_path
        with open(file_path, 'r') as file:
            existing_values = [line.strip() for line in file]

        print('Existing files :', len(existing_values))
        parameters = [value for value in parameters if (stock + '_candle_' + str(value[0]) +  '_strike_' + str(value[2])  + '_entry_' + str(value[3]).replace(':', ',') + '_exit_'  + '_' + start_date + '_' + end_date) not in existing_values]
        print('Parameters to run :', len(parameters))

        for value in  parameters:
            string =  (stock + '_candle_' + str(value[0]) +  '_strike_' + str(value[2])  + '_entry_' + str(value[3]).replace(':', ',') + '_exit_'  + '_' + start_date + '_' + end_date)
            print(string)
        
        # Start tradesheet generation
        start_time = time.time()
        # num_processes = multiprocessing.cpu_count()
        # #print('No. of processes :', num_processes)
        num_processes = 12
        print('No. of processes :', num_processes)

        partial_process = partial(parameter_process, mapped_days=mapped_days, option_data=option_data, df=df, start_date=start_date, end_date=end_date, counter=counter, output_folder_path=output_folder_path)
        with multiprocessing.Pool(processes = num_processes) as pool:
            
            with tqdm(total = len(parameters), desc = 'Processing', unit = 'Iteration') as pbar:
                def update_progress(combinations):
                    with open(txt_file_path, 'a') as fp:
                        line = str(combinations) + '\n'
                        fp.write(line)
                    pbar.update()
                
                arg_tuples = [tuple(parameter) for parameter in parameters]
                
                for result in pool.imap_unordered(partial_process, arg_tuples):
                    update_progress(result)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Time taken to get Initial Tradesheets:', elapsed_time)
print('Finished at :', time.time())