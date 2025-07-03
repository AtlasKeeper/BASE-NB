# import datetime as dt
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
import talib as ta
import time
from tqdm import tqdm
from datetime import datetime, timedelta
import os
from openpyxl import load_workbook
import ast
import json
from functools import partial
import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.insert(0, r"/home/newberry4/jay_data/")
from Common_Functions.utils import TSL_PREMIUM, postgresql_query, resample_data, nearest_multiple, round_to_next_5_minutes_d
from Common_Functions.utils import get_target_stoploss, get_open_range, check_crossover, compare_month_and_year

def postgresql_query(input_query, input_tuples = None):
    try:
        connection = psycopg2.connect(
            host = "algo-backtest-data-do-user-14334263-0.b.db.ondigitalocean.com",
            port = 25060,
            database = "defaultdb",
            user = "doadmin",
            password = "AVNS_kOwuGIv2gd1DmiPl9Cx",
            sslmode = "require"
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


def round_to_next_5_minutes(time_str):

    #print(time_str, type(time_str))
    # Parse the input time string to a datetime object
    time_str = time_str.strftime('%Y-%m-%d %H:%M:%S')

    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

    # Calculate the minutes to the next 5-minute multiple
    minutes_to_next_5 = (5 - time_obj.minute % 5) % 5

    # Round the time to the next 5-minute multiple
    rounded_time = time_obj + timedelta(minutes=minutes_to_next_5)

    # Format the rounded time as a string
    #rounded_time_str = rounded_time.strftime('%Y-%m-%d %H:%M:%S')
    return rounded_time


def get_target_stoploss(row, Type, Band, TARGET_TH, STOPLOSS_TH):
    
    Open = row['Open']
    High = row['High']
    Low = row['Low']
    Close = row['Close']
    
    # Stoploss Calculation
    # CE
    if Type == 'CE':
        if STOPLOSS_TH[0] == 'Band':
            stoploss = Low - Band*STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Points':
            stoploss = Low - STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Bips':
            stoploss = Low*(1 - 0.0001*STOPLOSS_TH[1])
        elif STOPLOSS_TH[0] == 'CandleLow':
            stoploss = Low
    # PE
    elif Type == 'PE':
        if STOPLOSS_TH[0] == 'Band':
            stoploss = High + Band*STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Points':
            stoploss = High + STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Bips':
            stoploss = High*(1 + 0.0001*STOPLOSS_TH[1])
        elif STOPLOSS_TH[0] == 'CandleLow':
            stoploss = High

    # Target Calculation
    # CE
    if Type == 'CE':
        if TARGET_TH[0] == 'Band':
            target = Close + Band*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleOC':
            target = Close + (Close - Open)*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleHL':
            target = Close + (High - Low)*TARGET_TH[1]
        elif TARGET_TH[0] == 'Bips':
            target = Close*(1 + 0.0001*TARGET_TH[1])
    
    #PE
    elif Type == 'PE':
        if TARGET_TH[0] == 'Band':
            target = Close - Band*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleOC':
            target = Close - (Open - Close)*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleHL':
            target = Close - (High - Low)*TARGET_TH[1]
        elif TARGET_TH[0] == 'Bips':
            target = Close*(1 - 0.0001*TARGET_TH[1])

    return target, stoploss


# Function to generate trade sheet
# def trade_sheet_creator(mapped_days, option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, MACD_SLOW, MACD_FAST, EMA_SLOW, EMA_FAST, LAST_ENTRY, STRIKE, I_T, L_P, INC_T, INC_S):
def trade_sheet_creator(mapped_days, option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, 
                            TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY ,TARGET_TH ,STOPLOSS_TH ,STRIKE,I_T ,L_P , INC_T ,INC_S):

    # Columns in the trade sheet
    column_names = ['Date', 'DateTime', 'Position', 'Time', 'Index Close', 'Target', 'Stoploss', 'StrikePrice', 'Action', 'Type', 'Exit', 'ExpiryDate', 'DaysToExpiry', 'Lots', 'Premium']
    trade_sheet = []
    
    # list of parameters in each combination
    target_list =  [TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY ,TARGET_TH ,STOPLOSS_TH ,STRIKE,I_T ,L_P , INC_T ,INC_S]

    # save what's being saved in new_text.txt
    # strategy_name = stock + "_" + str(TIME_FRAME) + "_" + Type + '_candle_' + 'FastEMA' + str(FAST_EMA) + "_SlowEMA" + str(SLOW_EMA) + \
    #                 "_ITM" + str(STRIKE) + "_ISL" + str(ISL) + "_TP" + str(TAKE_PROFIT) + \
    #                 "_InitialTarget" + str(I_T) + "_TargetINC" + str(INC_T) + "_StopLossINC" + str(INC_SL) + "_" + str(LAST_ENTRY) + "_TargetTH" + str(TARGET_TH)

    # sanitized_strategy_name = strategy_name.replace('.', ',').replace(':', ',')

    # Check for each combination for each dte if that combination needs to be processed for the current time period
    # We are storing for each combination for each dte if it's profitable or not in filter_df
    # So for counter = 1 we don't have to check this cause we'll run for all the combinations
    strategy_name = f'{stock}_candle_{TIME_FRAME}_fast_ema_{FAST_EMA}_slow_ema_{SLOW_EMA}_exit_{LAST_ENTRY}_target_{TARGET_TH[0]}_{TARGET_TH[1]}_stoploss_{STOPLOSS_TH[0]}_{STOPLOSS_TH[1]}_strike_{STRIKE}_IT_{I_T}_LP_{L_P}_INCT_{INC_T}_INCS_{INC_S}'
    sanitized_strategy_name = strategy_name.replace('.', ',').replace(':', ',')

    if counter!=1:
        filter_df_temp = filter_df[filter_df['Parameters'].apply(lambda x: x==str(target_list))].drop(columns=['Status'])
        #print(filter_df_temp)
        dte_to_process1 = filter_df_temp.columns[filter_df_temp.iloc[0] == 1].tolist()
        dte_to_process = [int(column[3:]) for column in dte_to_process1]
        #print(dte_to_process)
        
        mapped_days_temp = mapped_days[mapped_days['DaysToExpiry'].isin(dte_to_process)]   
    else:
        mapped_days_temp = mapped_days

    mapped_days_temp = mapped_days_temp[(mapped_days_temp['Date']>=start_date) & (mapped_days_temp['Date']<=end_date)]
    mapped_days_temp = mapped_days_temp[mapped_days_temp['Date']>'2021-06-03']

    # Iterate through each date to generate entry signals

    start_time1 = time.time()
    for _, row in mapped_days_temp.iterrows():
        
        date = row['Date']
        expiry_date = row['ExpiryDate']
        days_to_expiry = row['DaysToExpiry']

    
        # Ask Sourav about end_time
        start_time = pd.to_datetime(f'{date} 09:15:00')
        end_time = pd.to_datetime(f'{date} {LAST_ENTRY}:00')
        exit_time_period = pd.to_datetime(f'{date} 09:15:00')
        last_entry = end_time - pd.to_timedelta(TIME_FRAME)

        daily_data = resampled[(resampled.index >= start_time) & (resampled.index <= end_time)]
        
        if daily_data.empty:
            continue



        daily_data_start_time = pd.to_datetime(f'{date} 09:15:00') + pd.to_timedelta(TIME_FRAME)
        daily_data = resampled[(resampled.index >= daily_data_start_time) & (resampled.index <= end_time)]
        position = 0
        
        # Iterate through 5 min. data to generate entry signals
        for time_period, dd_row in daily_data.iloc[:-1].iterrows():
            
            # Bullish Entry : EMA_signal or MACD_signal
            if (position == 0) and (time_period > exit_time_period)and ((time_period + pd.to_timedelta(TIME_FRAME)) < last_entry) and \
                (dd_row['EMA_Signal'] == 1):
                
                position = 1
                #print(time_period, last_entry)
                next_time_period = time_period + pd.to_timedelta(TIME_FRAME)
                # next_index_open = daily_data.at[next_time_period, 'Open']
                
                index_close = daily_data.at[time_period, 'Close']
                print(index_close)
                index_low = daily_data.at[time_period, 'Low']
                ATM = nearest_multiple(index_close, roundoff)

                if Type == 'CE':
                    strike_price = ATM - STRIKE
                elif Type == 'PE':
                    strike_price = ATM + STRIKE


                print(strike_price)

                # if next_time_period not in option_data.index:
                #     print(f"No option data available for next_time_period: {next_time_period}")
                #     print(strike_price)
                #     continue
                
                # target, stoploss = get_target_stoploss(dd_row, Type, None, ('Bips', TARGET_TH), ('Bips', ISL))
                entry_price, exit_price, exit_time_period, exit_type, position, target, stoploss = TSL_PREMIUM(df, option_data, next_time_period, strike_price, 'long', Type, position, TARGET_TH[1], STOPLOSS_TH[1], LAST_ENTRY, I_T, L_P, INC_T, INC_S)

                exit_next_time_period = exit_time_period #+ pd.to_timedelta('1T')
                try:
                    exit_index_close = df.at[exit_next_time_period - pd.to_timedelta('1T'), 'Close']
                except KeyError:
                    next_index = df.index[df.index < (exit_next_time_period - pd.to_timedelta('1T'))].max()
                    exit_index_close = df.at[next_index, 'Close']

                trade_sheet.append(pd.Series([date, time_period, position, next_time_period, index_close, target, stoploss, strike_price, 'long', Type, '', expiry_date, days_to_expiry, 1, entry_price], index = column_names))
                trade_sheet.append(pd.Series([date, exit_time_period, position, exit_next_time_period, exit_index_close, '', '', strike_price, 'short', Type, exit_type, expiry_date, days_to_expiry, 1, exit_price], index = column_names))

                exit_time_period = round_to_next_5_minutes_d(exit_time_period, TIME_FRAME) 

    # try:
    #     #Trade sheet for each combination
    #     trade_sheet = pd.concat(trade_sheet, axis = 1).T
    # except Exception as e:
    #     # If there are no trades, we'll just return the name to be saved in the new_done.txt file
    #     print(e)
    #     return sanitized_strategy_name + '_' + start_date + '_' + end_date

    try:
        #Trade sheet for each combinatioEn
        trade_sheet = pd.concat(trade_sheet, axis = 1).T
    except Exception as e:
        # If there are no trades, we'll just return the name to be saved in the new_done.txt file
        print(e)
        return sanitized_strategy_name + '_' + start_date + '_' + end_date

    trade_sheet['Time'] = pd.to_datetime(trade_sheet['Time'])
    trade_sheet['Time'] = trade_sheet['Time'].dt.strftime('%H:%M:%S')

    

    #tradesheet.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', index = False)
    trade_sheet['Premium'] = np.where(trade_sheet['Action']=='long', -trade_sheet['Premium'], trade_sheet['Premium'])
    
    # create filter_df to store profitable combo and dte
    filter_df1 = pd.DataFrame(columns=['Strategy', 'Parameters', 'DTE0', 'DTE1', 'DTE2', 'DTE3', 'DTE4', 'Status'])
    filter_df1.loc[len(filter_df1), 'Strategy'] = sanitized_strategy_name
    row_index = filter_df1.index[filter_df1['Strategy'] == sanitized_strategy_name].tolist()[0]
    filter_df1.loc[row_index, 'Parameters'] = target_list
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 0
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Start_Date'] = start_date
    filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'End_Date'] = end_date
    
    # Go through each dte for the current combo to check if it's profitable
    for dte in dte_list:
        trade_sheet_temp = trade_sheet[trade_sheet['DaysToExpiry'] == dte]
        if not trade_sheet_temp.empty:
            trade_sheet_temp.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
            if trade_sheet_temp['Premium'].sum() > 0:
                filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 1
                filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 1
                # trade_sheet_temp = trade_sheet_temp[trade_sheet_temp['Date']>'2021-06-03']
                # trade_sheet_temp.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
            else:
                filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 0

    # Store the combo and it's dte which is profitable in filter_df file
    if filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'].iloc[0] == 1:
        
        existing_csv_file = f"{filter_df_path}/filter_df{counter}.csv"
        if os.path.isfile(existing_csv_file):
            filter_df1.to_csv(existing_csv_file, index=False, mode='a', header=False)
        else:
            filter_df1.to_csv(existing_csv_file, index=False)

    # print(filter_df1.head())
    end_time1 = time.time()
    # print('Time taken to run current combination :', (end_time1 - start_time1))
    return sanitized_strategy_name + '_' + start_date + '_' + end_date

# Function to calculate trailing stoploss
# def TSL(df, option_data, next_time_period, LAST_ENTRY, ATM, Action, Type, position, index_low, I_T, L_P, INC_T, INC_S):

# Function to calculate trailing stoploss
def TSL(df, option_data, next_time_period, ATM, Action, Type, position, target, stoploss, LAST_ENTRY, I_T, L_P, INC_T, INC_S):

    start_time = next_time_period
    end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')

    # Filter option data for the date, time, strike and type of the entry position    
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]
    intraday_data = intraday_data[(intraday_data['StrikePrice']==ATM) & (intraday_data['Type']==Type)]
    intraday_data = intraday_data.sort_index()
    entry_price = intraday_data.iloc[0]['Open']
    intraday_data = intraday_data[intraday_data.index > next_time_period]

    # Filter index data for the date, time, strike and type of the entry position
    temp_df  = df[(df.index > next_time_period) & (df.index < end_time)]
    
    if Action=='long':    
        target_pt = entry_price + I_T
        initial_stoploss_pt =  stoploss
        initial_target_pt = target
    target_hit = 0

    # crosses_threshold1 - when index goes below the initial stoploss pt (we take exit here)
    # crosses_threshold2 - when option premium goes above the profit pt
    # crosses_threshold3 - when option premium goes below the stoploss pt (we take exit here)
    # crooses_threshold4 - when index goes above the initial target pt (we take exit here)
    while True:
        
        # If the target pt is not hit once yet check for intitial stoploss pt hit
        if target_hit==0:
            if Type=='CE':
                crosses_threshold1 = (temp_df['Close'] < initial_stoploss_pt)
            elif Type=='PE':
                crosses_threshold1 = (temp_df['Close'] > initial_stoploss_pt)
        else:
            crosses_threshold1 = pd.Series()
         
        crosses_threshold2 = (intraday_data['Open'] > target_pt)

        if Type=='CE':
            crosses_threshold4 = (temp_df['Close'] > initial_target_pt)
        elif Type=='PE':
            crosses_threshold4 = (temp_df['Close'] < initial_target_pt)

        # print(next_time_period)
        if (crosses_threshold4.empty) & (temp_df.empty):
            crosses_threshold4 = pd.Series([False], index=[end_time])
            crosses_threshold4.index.name = 'DateTime'
        
        # If the target pt is hit already check for stoplos pt hit
        if target_hit:
            crosses_threshold3 = (intraday_data['Open'] < stoploss_pt)
        else:
            crosses_threshold3 = pd.Series()
        
        # When target pt is not hit for the first time yet and initial stoploss pt is also not hit throughout day
        if (target_hit==0) & (not crosses_threshold1.any()) & (not crosses_threshold2.any()) & (not crosses_threshold4.any()):
            exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
            exit_price = intraday_data.at[exit_time_period, 'Open']
            exit_type = 'Universal'
            position = 0
            break

        # When the target pt is not hit for the first time yet
        if (target_hit==0): 

            # When the initial stoploss pt got hit before target pt
            if crosses_threshold1.any() & crosses_threshold2.any() &  crosses_threshold4.any() & ((crosses_threshold1.idxmax()) < crosses_threshold2.idxmax()) \
                                                                                               & ((crosses_threshold1.idxmax()) < crosses_threshold4.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & crosses_threshold2.any() &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) < crosses_threshold1.idxmax()) \
                                                                                               & ((crosses_threshold4.idxmax()) <= crosses_threshold2.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break
            
            # Wenh the target pt got hit before the initial stoploss pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold1.idxmax()) \
                                                                                                & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax())    :
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
            
            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & (not crosses_threshold4.any()) & ((crosses_threshold1.idxmax()) < crosses_threshold2.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break


            # When the target pt got hit before the initial stoploss pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & (not crosses_threshold4.any()) & (crosses_threshold1.idxmax() > crosses_threshold2.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 3 : ', temp_df)

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & (not crosses_threshold2.any()) &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) < crosses_threshold1.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & (not crosses_threshold2.any()) &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) > crosses_threshold1.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif (not crosses_threshold1.any()) & crosses_threshold2.any() & crosses_threshold4.any() & ((crosses_threshold2.idxmax()) >= crosses_threshold4.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # WHen the target pt got hit before the initial stoploss pt
            elif (not crosses_threshold1.any()) & crosses_threshold2.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 4 : ', temp_df)


            # When the initial stoploss pt got hit but target pt is not hit
            elif crosses_threshold1.any():
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break
            
            # When the initial stoploss pt got hit but target pt is not hit
            elif crosses_threshold4.any():
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break
            
            # When only target pt got hit
            elif crosses_threshold2.any():
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 5 : ', temp_df)
                # print('Crosses_Threshold2 :', crosses_threshold2.idxmax())
                # print('Intraday_Data :', intraday_data)


        # When the target pt is already hit once
        elif target_hit: 
            
            #print('Crosses_Threshold4 : ', crosses_threshold4.any())
            # When target pt is already hit once and if target pt and stoploss pt none of them got hit throughout day take an exit
            # try:
            if (not crosses_threshold2.any()) & (not crosses_threshold3.any()) & (not crosses_threshold4.any()):
                
                exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                exit_price = intraday_data.at[exit_time_period, 'Open']
                exit_type = 'Universal'
                position = 0
                break 
            
            # When target pt is already hit once and stoploss pt got hit before target pt 
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & \
                (crosses_threshold3.idxmax() < crosses_threshold2.idxmax()) &  (crosses_threshold3.idxmax() < crosses_threshold4.idxmax()) :
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break


            # When target pt is already hit once and stoploss pt got hit before target pt 
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & (crosses_threshold4.idxmax() <= crosses_threshold2.idxmax())\
                                                                                                &  (crosses_threshold4.idxmax() < crosses_threshold3.idxmax())   :
                #print(ATM, exit_time_period)
                #print(intraday_data)
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            
            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold3.idxmax()) \
                                                                                                &  (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 6 : ', temp_df)


            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif (not crosses_threshold2.any()) & crosses_threshold3.any() & (crosses_threshold4.any()) & (crosses_threshold3.idxmax() < crosses_threshold4.idxmax()):
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif (not crosses_threshold2.any()) & crosses_threshold3.any() & (crosses_threshold4.any()) & (crosses_threshold4.idxmax() < crosses_threshold3.idxmax()):
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold2.any() & crosses_threshold3.any() & (not crosses_threshold4.any()) & (crosses_threshold3.idxmax() < crosses_threshold2.idxmax()):
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break

            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & crosses_threshold3.any() & (not crosses_threshold4.any()) & (crosses_threshold2.idxmax() < crosses_threshold3.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 7 : ', temp_df)



            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold2.any() & (not crosses_threshold3.any()) & crosses_threshold4.any() & (crosses_threshold4.idxmax() <= crosses_threshold2.idxmax()):
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break  

            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & (not crosses_threshold3.any()) & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 8 : ', temp_df)


            # When target pt is already hit once and only target pt got hit 
            elif crosses_threshold2.any():
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 9 : ', temp_df)



            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold3.any():
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break
            
            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold4.any():
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break
            # except:
            #     print('Crosses_Threshold4 ', crosses_threshold4)
            #     print('Temp df ', temp_df)
            #     print('Target List : ', target_list)
            #     exit()
        if position == 0:
            break

    return [entry_price, exit_price, exit_time_period, exit_type, position]


# Function to generate indicator signals bassed on the parameters
def parameter_process(parameter, mapped_days, option_data, df, filter_df, start_date, end_date, counter, output_folder_path):

    # TIME_FRAME, MACD_SLOW, MACD_FAST, EMA_SLOW, EMA_FAST, LAST_ENTRY, STRIKE, I_T, L_P, INC_T, INC_S = parameter

    TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY ,TARGET_TH ,STOPLOSS_TH ,STRIKE,I_T ,L_P , INC_T ,INC_S  = parameter
    
    resampled_df = resample_data(df,TIME_FRAME) 
    ## resample dataframe in index before hand.
    # resampled_df = resampled_df_main.copy()
    resampled = resampled_df.dropna()

    resampled_df['Fast_EMA'] = ta.EMA(resampled_df['Close'], timeperiod = FAST_EMA)
    resampled_df['Slow_EMA'] = ta.EMA(resampled_df['Close'], timeperiod = SLOW_EMA)

    resampled_df['Candle_Size'] = resampled_df['High'] - resampled_df['Low']
    # CE
    if Type == 'CE':
        resampled_df['EMA_Signal'] = np.where(
    (resampled_df['Fast_EMA'] > resampled_df['Slow_EMA']) & 
    (resampled_df['Fast_EMA'].shift(1) <= resampled_df['Slow_EMA'].shift(1)) &
    (resampled_df['Close'] > resampled_df['Fast_EMA']) &
    (resampled_df['Close'] > resampled_df['Slow_EMA']) &
    (resampled_df['Close'].shift(-1) > resampled_df['Open'].shift(-1)) &
    (resampled_df['Candle_Size'] < 40), 1, 0)
    # else:
    # If option type = 'PE'
    #     resampled_df['EMA_Signal'] = np.where(
    # (resampled_df['Fast_EMA'] > resampled_df['Slow_EMA']) & 
    # (resampled_df['Fast_EMA'].shift(1) <= resampled_df['Slow_EMA'].shift(1)) &
    # (resampled_df['Close'] > resampled_df['Fast_EMA']) &
    # (resampled_df['Close'] > resampled_df['Slow_EMA']) &
    # (resampled_df['Close'].shift(-1) > resampled_df['Open'].shift(-1)) &
    # (resampled_df['Candle_Size'] < 40), 1, 0)
    
    resampled_df['EMA_Signal'] = resampled_df['EMA_Signal'].shift(1)
    resampled = resampled_df.dropna()

    return trade_sheet_creator(mapped_days, option_data, df, filter_df, start_date, end_date, counter, output_folder_path, resampled, 
                            TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY ,TARGET_TH ,STOPLOSS_TH ,STRIKE,I_T ,L_P , INC_T ,INC_S)


#################################### Inputs ######################################################
superset = 'EMA_Crossover_CE'
stock = 'FINNIFTY'
Type = 'CE'

roundoff = 50 if stock == 'NIFTY' or stock == 'FINNIFTY' else (100 if stock == 'BANKNIFTY' else None)
crossover = 'Upper' if Type == 'CE' else ('Lower' if Type == 'PE' else None)

# Define all the file paths
root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{Type}/"
filter_df_path = rf"{root_path}/Filter_Sheets/"
# option_data_path = rf"/home/newberry4/jay_data/Data/{stock}/Current_Expiry/"
expiry_file_path = rf"/home/newberry4/jay_data/Common_Files/{stock} market dates.xlsx"
txt_file_path = rf'{root_path}/new_done.txt'
output_folder_path = rf'{root_path}/Trade_Sheets/'

if stock == 'NIFTY':
    option_data_path = rf"/home/newberry4/jay_data/Data/{stock}/Current_Expiry/"
elif stock =='BANKNIFTY':
    option_data_path = rf"/home/newberry4/jay_data/BANKNIFTY_DATA/BANKNIFTY_OHLCV/"
else:
    option_data_path = rf"/home/newberry4/jay_data/FINNIFTY_2/"

# Create all the required directories
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None

# list of period buckets
# date_ranges = [('2023-06-01', '2023-08-30')]


# date_ranges = [ ('2024-02-01', '2024-05-31'),
#                 ('2023-10-01', '2024-01-31'),
#                 ('2023-06-01', '2023-09-30'),
#                 ('2023-02-01', '2023-05-31'),
#                 ('2022-10-01', '2023-01-31'),
#                 ('2022-06-01', '2022-09-30'),
#                 ('2021-06-01', '2022-05-31')]

# #finnifty data range
# ('2024-02-01', '2024-03-20'),
date_ranges = [
                ('2023-10-01', '2024-05-31'),
                ('2023-06-01', '2023-09-30'),
                ('2023-01-01', '2023-05-31'),
              
            ]


# SIGNAL_PERIOD = 9
dte_list = [0, 1, 2, 3, 4]


#basic parameters best ******************************
candle_time_frames = ['3T','5T']
fast_emas = [3,5,10] 
slow_emas = [20,30,40] 
stoploss_thresholds = [('Pts', 20), ('Pts', 30), ('Pts', 40)] # ISL
target_thresholds = [('Pts', 25), ('Pts', 50), ('Pts', 75), ('Pts', 100)]
strikes = [0,50,100]
initial_target = [20,40]
lockin_profit = [0]
inc_target = [4]
inc_stoploss = [2]
last_entries = ['15:00']


#test paramters

# candle_time_frames = ['3T']
# fast_emas = [3] 
# slow_emas = [20] 
# stoploss_thresholds = [('Pts', 20)] # ISL
# target_thresholds = [('Pts', 25)]
# strikes = [0]
# initial_target = [20]
# lockin_profit = [0]
# inc_target = [4]
# inc_stoploss = [2]
# last_entries = ['15:00']


# Change parameters based on stock
if stock == 'BANKNIFTY':
    strikes = [x * 2 for x in strikes]
    initial_target = [30, 50]
    lockin_profit = [0,10]
    stoploss_thresholds = [('Pts', 30) , ('Pts', 50)] 
    target_thresholds = [('Pts', 60), ('Pts', 100), ('Pts', 125), ('Pts', 150)]

#############################################################################################

if __name__ == "__main__":

    counter = 0

    start_date_idx = date_ranges[-1][0]
    end_date_idx = date_ranges[0][-1]
    
    # Read expiry file
    mapped_days = pd.read_excel(expiry_file_path)
    mapped_days = mapped_days[(mapped_days['Date'] >= start_date_idx) & (mapped_days['Date'] <= end_date_idx)] #######
    weekdays = mapped_days['Date'].to_list()
    
    # Pull Index data
    start_time = time.time()

    current_time = datetime.now().strftime("%H:%M:%S")

    # Print the current time
    print("Start Time : ", current_time)

    print(start_date_idx, end_date_idx, superset, stock, Type)


    table_name = stock + '_IDX'
    if table_name:
        print("suceess")
    
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
    # df = df[(df['DaysToExpiry'] == 0) | (df['DaysToExpiry'] == 1)]          #### added for finnifty
    print(df.head(4))
    print(df.iloc[0]) 
    # resampled_df_main = resample_data(df, '5T')   ######

    for start_date, end_date in date_ranges: 

        counter += 1
        print(start_date, end_date, counter)
        print(superset, stock, Type)

        # output_folder_path = f'/home/newberry/EMA Crossover copy/Trade Sheets/{stock}/{Type}/'
        # start_date = '2024-02-01'
        # end_date = '2024-05-20'
        # Pull Options data
        start_time = time.time()
        # option_data_path = rf"/home/newberry/Global Files/Options Data/{stock}/"
        option_data_files = next(os.walk(option_data_path))[2]
        option_data = pd.DataFrame()

        for file in option_data_files:
            file1 = compare_month_and_year(start_date, end_date, file, stock)
                
            if not file1:
                continue

            temp_data = pd.read_pickle(option_data_path + file)[['ExpiryDate', 'StrikePrice', 'Type', 'Open', 'Ticker']]
            temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
            temp_data = temp_data.rename_axis('DateTime')
            option_data = pd.concat([option_data, temp_data])

        option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
        option_data['Type'] = option_data['Type'].astype('category')
        option_data = option_data.sort_index()
        end_time = time.time()
        print('Time taken to pull Options data :', (end_time-start_time))

        
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
   
            for TIME_FRAME in candle_time_frames:
                for FAST_EMA in fast_emas:
                    for SLOW_EMA in slow_emas:
                        for LAST_ENTRY in last_entries:
                            for TARGET_TH in target_thresholds:
                                for STOPLOSS_TH in stoploss_thresholds:
                                    for STRIKE in strikes:
                                        for I_T in initial_target:
                                            for L_P in lockin_profit:
                                                if L_P <= I_T:
                                                    for INC_T in inc_target:
                                                        for INC_S in inc_stoploss:
                                                            if INC_S <= INC_T and STOPLOSS_TH <= TARGET_TH:

                                                                if (INC_S==1) & (INC_T==4):
                                                                    continue 
                                                                if (FAST_EMA < SLOW_EMA):
                                                                    parameter_combination = [TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY, TARGET_TH, STOPLOSS_TH, STRIKE, \
                                                                                            I_T, L_P, INC_T, INC_S]
                                                                    
                                                                    parameters.append(parameter_combination)

        # parameters = parameters[:200]

        # Read the content of the log file to check which parameters have already been processed
        file_path = txt_file_path
        with open(file_path, 'r') as file:
            existing_values = [line.strip() for line in file]


        print('Total Parameters : ', len(parameters))   ######
        print('Existing Parameters : ', len(existing_values))

        # parameters = [TIME_FRAME, RSI_THRESHOLD, RSI_WINDOW, STRIKE, ISL, 
        #               TAKE_PROFIT, I_T, INC_T, INC_SL, LAST_ENTRY]
        # parameters = [value for value in parameters if (stock + "_" + str(value[0]) + '_candle_' + 'RSI' + str(value[1]) + "_W" + str(value[2]) +
        #                                                 "_ITM" + str(value[3]) + "_ISL" + str(value[4]) + "_TP" + str(value[5 ]) +
        #                                                 "_TargetINC" + str(value[6]) + "_StopLossINC" + str(value[7]) + "_" + str(value[8])) not in existing_values]
        
        # [TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY, TARGET_TH, STOPLOSS_TH, STRIKE, I_T, L_P, INC_T, INC_S]
        # if stock == 'BANKNIFTY':
        # parameters = [value for value in parameters if (stock + "_" + Type + '_candle_'  + str(value[0]) + 'fast_ema' + str(value[1]) + "_slow_ema" + str(value[2]) +
        #                                                 '_exit_' + str(value[3]) + "_target_" + str(value[4][0]) + '_' + str(value[4][1]) + "_stoploss_" + str(value[5][0]) + '_' + str(value[5][1]) + "_strike" + str(value[6]) + "_IT_" + str(value[7]) + "_LP_" + str(value[8]) +
        #                                                 "_INCT_" + str(value[9]) + "_INCS" + str(value[10]) + "_" + start_date + "_" + end_date).replace(".", ",")\
        #                                             not in existing_values]  #####
        
        parameters = [value for value in parameters if (stock + '_candle_' + str(value[0]) +  '_fast_ema_' + str(value[1]) + "_slow_ema_" + str(value[2]) + '_exit_' + str(value[3]).replace(':', ',') + \
                                                        '_target_' + str(value[4][0]) + '_' + str(value[4][1]).replace('.', ',') + '_stoploss_' + \
                                                        str(value[5][0]) + '_' + str(value[5][1]).replace('.', ',') + '_strike_' + str(value[6]) + \
                                                        '_IT_' + str(value[7]) + '_LP_' + str(value[8]) + '_INCT_' + str(value[9]).replace('.', ',') + '_INCS_' + \
                                                        str(value[10]).replace('.', ',') + '_' +  start_date + '_' + end_date) not in existing_values]
        
        # for value in  parameters:
        #     string =  (stock + '_candle_' + str(value[0]) +  '_fast_ema_' + str(value[1]) + "_slow_ema_" + str(value[2]) + '_exit_' + str(value[3]).replace(':', ',') + \
        #                                                 '_target_' + str(value[4][0]) + '_' + str(value[4][1]).replace('.', ',') + '_stoploss_' + \
        #                                                 str(value[5][0]) + '_' + str(value[5][1]).replace('.', ',') + '_strike_' + str(value[6]) + \
        #                                                 '_IT_' + str(value[7]) + '_LP_' + str(value[8]) + '_INCT_' + str(value[9]).replace('.', ',') + '_INCS_' + \
        #                                                 str(value[10]).replace('.', ',') + '_' +  start_date + '_' + end_date)
        #     print(string)
        # # else:
        #     parameters = [value for value in parameters if (stock + "_" + str(value[0]) + "_" + Type + '_candle_' + 'FastEMA' + str(value[1]) + "_SlowEMA" + str(value[2]) +
        #                                                     "_ITM" + str(value[3]) + "_ISL" + str(value[4]) + "_TP" + str(value[5]) + "_InitialTarget" + str(value[6]) +
        #                                                     "_TargetINC" + str(value[7]) + "_StopLossINC" + str(value[8]) + "_" + str(value[9].replace(":", ",")) + "_TargetTH" + str(value[10]) + "_" + start_date + "_" + end_date)\
        #                                                 not in existing_values]

        print('Remaining Parameters : ', len(parameters))

        # resampled_df_main = resample_data(df, '5T')
        # Start tradesheet generation
        start_time = time.time()
        #num_processes = multiprocessing.cpu_count()
        # #print('No. of processes :', num_processes)
        num_processes = 16
        print('No. of processes :', num_processes)
        
        # partial_process = partial(parameter_process, resampled_df_main=resampled_df_main, mapped_days=mapped_days, option_data=option_data, df=df, filter_df=filter_df, start_date=start_date, end_date=end_date, counter=counter, output_folder_path=output_folder_path)
        partial_process = partial(parameter_process, mapped_days=mapped_days, option_data=option_data, df=df, filter_df=filter_df, start_date=start_date, end_date=end_date, counter=counter, output_folder_path=output_folder_path) 
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
        end_time = datetime.now().strftime("%H:%M:%S")

        # Print the current times
        print("End Time : ", end_time)
