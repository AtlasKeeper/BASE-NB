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
sys.path.insert(0, r"/home/newberry3/user/")
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
    """
    Resamples financial time series data to a specified time frame and aggregates OHLC values.
    Parameters:
        data (pd.DataFrame): Input DataFrame with a DateTimeIndex and columns ['Open', 'High', 'Low', 'Close', 'ExpiryDate'].
        TIME_FRAME (str): Pandas-compatible resampling frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour).
    Returns:
        pd.DataFrame: Resampled DataFrame with aggregated OHLC values, 'ExpiryDate', and additional 'Date' and 'Time' columns.
    """
    
    resampled_data = data.resample(TIME_FRAME).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data

def nearest_multiple(x, n):
    """
    Returns the nearest multiple of `n` to the given number `x`.
    If `x` is exactly halfway between two multiples of `n`, rounds up to the higher multiple.

    Args:
        x (int or float): The number to round.
        n (int): The multiple to which to round.

    Returns:
        int: The nearest multiple of `n` to `x`.
    """
    remainder = x%n
    if remainder < n/2:
        nearest = x - remainder
    else:
        nearest = x - remainder + n
    return int(nearest)

def get_strike(ATM, minute, daily_option_data):
    
    def get_premium(option_type):
        """
        Retrieves the premium (open price) for a given option type at the at-the-money (ATM) strike price.
        Args:
            option_type (str): The type of the option ('CE' for Call, 'PE' for Put).
        Returns:
            float or None: The open price (premium) of the option if available, otherwise None.
        """
        
        subset = temp_option_data[(temp_option_data['StrikePrice'] == ATM) & (temp_option_data['Type'] == option_type)]
        
        if subset.empty:
            return None
        
        return subset['Open'].iloc[0]
    
    def find_nearest_strike(target_premium, option_type):
        """
        Finds the strike price and premium of the option whose 'Open' price is nearest to the target premium for a given option type.
        Args:
            target_premium (float): The target premium to find the nearest match for.
            option_type (str): The type of option to filter by (e.g., 'CE' for call, 'PE' for put).
        Returns:
            tuple: A tuple containing:
                - nearest_strike (float or None): The strike price of the option with the nearest premium, or None if no match is found.
                - nearest_premium (float or None): The premium ('Open' price) of the nearest option, or None if no match is found.
        """
        
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
    """
    Calculates the final premium values for a set of option trades by merging premium data with option price data,
    applying lot size and brokerage adjustments, and computing the net premium for each trade.
    Args:
        premium_data (pd.DataFrame): DataFrame containing trade information such as Date, Time, ExpiryDate, ATM, CE_OTM, PE_OTM, and Position.
        option_data (pd.DataFrame): DataFrame containing option price data with columns including Ticker, Type ('CE' or 'PE'), StrikePrice, and Open.
        RATIO (tuple): A tuple (long_lots, short_lots) specifying the number of lots for long and short positions.
    Returns:
        pd.DataFrame: The input premium_data DataFrame augmented with calculated premium columns for each leg (CE_ATM, PE_ATM, CE_OTM, PE_OTM),
                      the total Premium, and DaysToExpiry.
    """
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
            """
            Loads and concatenates options data from pickled files within a specified date range and for a specific stock.
            Args:
                start_date (str or datetime): The start date for filtering option data files.
                end_date (str or datetime): The end date for filtering option data files.
                option_data_path (str): The directory path where option data files are stored.
                stock (str): The stock ticker symbol to filter relevant option data files.
            Returns:
                pd.DataFrame: A DataFrame containing concatenated options data filtered by date and stock, 
                              with columns ['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open', 'High', 'Low', 'Close', 'Ticker'].
                              The DataFrame index is set to the parsed datetime from the 'Ticker' column and named 'DateTime'.
            Notes:
                - Only files matching the month and year criteria (as determined by compare_month_and_year) are loaded.
                - The function prints the columns of the resulting DataFrame and the time taken to load the data.
                - The 'StrikePrice' column is cast to int32 and 'Type' to category for memory efficiency.
            """
            
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
    """
    Retrieves index data for a given stock between specified start and end date indices.
    This function queries a PostgreSQL database for OHLC (Open, High, Low, Close) data and ticker information
    for the specified stock within the provided date range and during trading hours (09:15 to 15:29).
    The resulting data is merged with a mapped_days DataFrame, indexed by datetime, sorted, and duplicates are removed.
    Args:
        start_date_idx (str): The start date index in 'YYYYMMDD' format.
        end_date_idx (str): The end date index in 'YYYYMMDD' format.
        stock (str): The stock symbol for which to retrieve index data.
    Returns:
        pandas.DataFrame: A DataFrame containing the merged and processed index data, indexed by datetime.
    """

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
        """
        Generates a trade sheet for a given options backtesting strategy over a specified date range.
        This function processes mapped trading days, filters them based on date constraints, and simulates option trades
        according to the provided strategy parameters. It records entry and exit trades, calculates premiums, and saves
        the results to CSV files. Additionally, it maintains a filter DataFrame to track the profitability of different
        strategy parameter combinations.
        Parameters:
            mapped_days (pd.DataFrame): DataFrame containing trading days and expiry information.
            option_data (pd.DataFrame): DataFrame containing option price data indexed by datetime.
            df (pd.DataFrame): DataFrame containing index price data (e.g., NIFTY, BANKNIFTY) indexed by datetime.
            start_date (str): Start date for backtesting in 'YYYY-MM-DD' format.
            end_date (str): End date for backtesting in 'YYYY-MM-DD' format.
            counter (int): Counter used for naming output filter files.
            output_folder_path (str): Path to the folder where trade sheets will be saved.
            resampled (bool): Indicates if the data is resampled (unused in this function).
            TIME_FRAME (int): Candle time frame in minutes for the strategy.
            STRIKE (int): Strike difference for selecting option contracts.
            ENTRY (str): Entry time for trades in 'HH:MM' format.
            EXIT (str): Exit time for trades in 'HH:MM' format.
        Returns:
            str: A string containing the sanitized strategy name along with the start and end dates, 
                 used as a unique identifier for the generated trade sheet.
        Notes:
            - The function expects certain global variables and helper functions (e.g., `nearest_multiple`, `TSL_SS`, `stock`, `filter_df_path`) to be defined elsewhere.
            - The function writes trade sheets and filter DataFrames to CSV files in the specified output directory.
            - Some features, such as re-entry logic and DTE-based profitability checks, are present but commented out.
        """
    
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
    """
    Simulates a trailing stop-loss (TSL) strategy for options trading, evaluating entry and exit conditions for both Call (CE) and Put (PE) options.

    Parameters:
        option_data (pd.DataFrame): DataFrame containing intraday option data with columns including 'StrikePrice', 'Type', 'Open', 'High', 'Close', and a datetime index.
        next_time_period (datetime): The datetime at which the trade is initiated.
        CE_OTM (float or int): Strike price for the Out-of-the-Money Call Option (CE).
        PE_OTM (float or int): Strike price for the Out-of-the-Money Put Option (PE).
        EXIT (str): The exit time as a string in 'HH:MM' format.
        days_to_expiry (int): Number of days remaining until the option's expiry.

    Returns:
        list: A list containing:
            - CE_OTM_entry_price (float): Entry price for the CE leg.
            - CE_OTM_exit_price (float): Exit price for the CE leg.
            - exit_time (datetime): Time at which the exit condition was met.
            - PE_OTM_entry_price (float): Entry price for the PE leg.
            - PE_OTM_exit_price (float): Exit price for the PE leg.
            - exit_time (datetime): Time at which the exit condition was met.
            - status (str): Description of the exit condition ("No CE Data", "No PE Data", "No Exit Data", "Time Exit", etc.).

    Notes:
        - If no data is available for the specified CE or PE strike, returns zeros and a status message.
        - The function currently only implements a time-based exit; stop-loss and target logic is commented out.
        - Assumes the input DataFrame is indexed by datetime and contains the necessary columns.
    """
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
    """
    Implements a trailing stop loss (TSL) strategy for options trading, handling both call (CE) and put (PE) options.
    The function tracks intraday price movements for specified option strikes, applies stop loss and target logic,
    and determines exit points based on price thresholds or end-of-day exit time.
    Parameters:
        option_data (pd.DataFrame): DataFrame containing intraday option data with a DateTimeIndex and columns including 'StrikePrice', 'Type', and 'Open'.
        next_time_period (pd.Timestamp): The entry time for the trade.
        CE_OTM (float or int): Strike price for the out-of-the-money call option (CE).
        PE_OTM (float or int): Strike price for the out-of-the-money put option (PE).
        EXIT (str): Time (in 'HH:MM' format) to force exit the trade if stop loss/target is not hit.
        STOPLOSS_PT (float): Stop loss percentage (e.g., 0.25 for 25%).
        PREMIUM_TP (str): Determines which premium to use for target calculation ('MIN' for minimum of CE/PE, otherwise uses respective premiums).
    Returns:
        list: [
            CE_OTM_entry_price (float): Entry price for CE option,
            CE_OTM_exit_price (float): Exit price for CE option,
            ce_exit_time_period (pd.Timestamp): Exit time for CE option,
            PE_OTM_entry_price (float): Entry price for PE option,
            PE_OTM_exit_price (float): Exit price for PE option,
            pe_exit_time_period (pd.Timestamp): Exit time for PE option,
            ce_stoploss (int): 1 if CE stop loss/target hit, 0 if exited at end time,
            pe_stoploss (int): 1 if PE stop loss/target hit, 0 if exited at end time,
            CE_OTM_new_entry_price (float or str): New entry price for CE if re-entered after PE stop loss, else '',
            PE_OTM_new_entry_price (float or str): New entry price for PE if re-entered after CE stop loss, else ''
        ]
    Notes:
        - The function assumes 1-minute frequency data.
        - If the stop loss/target is not hit, the position is exited at the specified EXIT time.
        - Handles cases where price data may be missing at exact exit times by using the last available price before the exit time.
    """
    
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
    """
    Resamples financial time series data to a specified time frame and aggregates OHLC values.
    Parameters:
        data (pd.DataFrame): Input DataFrame with a DateTimeIndex and columns ['Open', 'High', 'Low', 'Close', 'ExpiryDate'].
        TIME_FRAME (str): Pandas-compatible resampling frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour).
    Returns:
        pd.DataFrame: Resampled DataFrame with aggregated OHLC values, 'ExpiryDate', and additional 'Date' and 'Time' columns.
    """
    
    resampled_data = data.resample(TIME_FRAME).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data

########################################### INPUTS #####################################################

# Basic configuration for the strategy
superset = 'plain_vanilla'              # Strategy category/folder name
stock = 'NIFTY'                         # Stock/index being tested
option_type = 'ND'                      # Option category (e.g., 'ND' could mean non-directional)

# Set roundoff step, brokerage, and lot size based on selected stock
roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' or stock == 'SENSEX' else None)
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' or stock == 'SENSEX' else None)
LOT_SIZE = 25 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else 10)

# Define folder structure based on parameters
root_path = rf"/home/newberry3/user/STARTEGY_NAME/{superset}/{stock}/{option_type}/"
filter_df_path = rf"{root_path}/Filter_Sheets/"                   # Folder for filtered parameter files
expiry_file_path = rf"/home/newberry3/user/Common_Files/{stock} market dates.xlsx"   # Excel containing expiry mapping
txt_file_path = rf'{root_path}/new_done.txt'                     # File to track completed parameters
output_folder_path = rf'{root_path}/Trade_Sheets/'               # Folder for final trade sheets

# Set option data path depending on stock type
if stock == 'NIFTY':
    option_data_path = rf"/home/newberry3/user/Data/NIFTY/folder/folder/"
elif stock == 'BANKNIFTY':
    option_data_path = rf"/home/newberry3/user/Data/BANKNIFTY/folder/"
elif stock == 'FINNIFTY':
    option_data_path = rf"/home/newberry3/user/Data/FINNIFTY/folder/"
elif stock == 'SENSEX':
    option_data_path = rf"/home/newberry3/user/Data/SENSEX/folder/"

# Ensure necessary folders and tracking file exist
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None

# Define the backtesting date ranges
date_ranges = [ 
    ('2024-06-01', '2024-10-30'),
    ('2024-02-01', '2024-05-31'),
    ('2023-10-01', '2024-01-31'),
    ('2023-06-01', '2023-09-30'),
    ('2023-02-01', '2023-05-31'),
    ('2022-10-01', '2023-01-31'),
    ('2022-06-01', '2022-09-30'),
    ('2022-01-01', '2022-05-31'), 
    ('2021-06-01', '2021-12-31')
]

# Strategy parameters
candle_time_frame = ['5T']          # 5-minute candle frequency
entries = ['09:25']                 # Entry time
exits = ['15:20']                   # Exit time
strikes = [0]                       # Strike offset (0 = ATM)

parameters = []                     # To store all parameter combinations

# Function that handles trade logic for each parameter set
def parameter_process(parameter, mapped_days, option_data, df, start_date, end_date, counter, output_folder_path):
    TIME_FRAME, STRIKE, ENTRY, EXIT = parameter
    resampled_df = resample_data(df, TIME_FRAME)         # Resample index data to desired timeframe
    resampled = resampled_df.dropna()                    # Drop missing values
    return trade_sheet_creator(mapped_days, option_data, df, start_date, end_date,
                               counter, output_folder_path, resampled, TIME_FRAME, STRIKE, ENTRY, EXIT)

# Adjust strike step for BANKNIFTY or SENSEX (e.g., step size is 2x)
if stock == 'BANKNIFTY' or stock == 'SENSEX':
    strikes = [x * 2 for x in strikes]

# Main execution block
if __name__ == "__main__":
    
    counter = 0
    start_date_idx = date_ranges[-1][0]     # Start from earliest range
    end_date_idx = date_ranges[0][-1]       # End at latest range

    # Read expiry date file and filter between the start/end date
    mapped_days = pd.read_excel(expiry_file_path)
    mapped_days = mapped_days[(mapped_days['Date'] >= start_date_idx) & (mapped_days['Date'] <= end_date_idx)]
    mapped_days = mapped_days.rename(columns={'WeeklyDaysToExpiry' : 'DaysToExpiry'})  # Rename for uniformity

    # Pull index price data for the backtesting window
    df = pull_index_data(start_date_idx, end_date_idx, stock)
    resampled_df_main = resample_data(df, '5T')          # Resample entire index data once

    # Iterate over each date range for backtest
    for start_date, end_date in date_ranges: 
        counter += 1
        print(start_date, end_date, counter)

        start_date_object = pd.to_datetime(start_date)
        end_date_object = pd.to_datetime(end_date)
        new_end_date_object = end_date_object + timedelta(days=30)     # To capture expiry after range
        new_end_date = new_end_date_object.strftime('%Y-%m-%d')        # Convert to string

        # Pull relevant options data for the extended window
        option_data = pull_options_data_d(start_date, new_end_date, option_data_path, stock)

        parameters = []        # Reset parameters list

        if counter == 1:
            filter_df = pd.DataFrame()      # For the first run, no filters
        elif counter > 1:
            # For subsequent runs, load prior filter data
            filter_file = f"{filter_df_path}/filter_df{counter-1}.csv"
            if not os.path.exists(filter_file):
                print(f"File filter_df{counter-1}.csv does not exist. Stopping the code.")
                sys.exit()
            else:
                filter_df = pd.read_csv(filter_file)
                filter_df = filter_df.drop_duplicates()

        # Read previous run parameters
        if counter != 1:
            parameters = filter_df['Parameters'].to_list()
            parameters = [ast.literal_eval(item.replace("'", "\"")) for item in parameters]

        # For first run, generate all parameter combinations
        elif counter == 1:
            for TIME_FRAME in candle_time_frame:
                for STRIKE in strikes:
                    for ENTRY in entries:
                        for EXIT in exits:
                            if ENTRY < EXIT:
                                parameters.append([TIME_FRAME, STRIKE, ENTRY, EXIT])

        # Filter out parameters that are already processed (tracked in txt file)
        print('Total parameters :', len(parameters))
        file_path = txt_file_path
        with open(file_path, 'r') as file:
            existing_values = [line.strip() for line in file]

        print('Existing files :', len(existing_values))
        parameters = [value for value in parameters if
                      (stock + '_candle_' + str(value[0]) + '_strike_' + str(value[2]) +
                       '_entry_' + str(value[3]).replace(':', ',') +
                       '_exit_' + '_' + start_date + '_' + end_date) not in existing_values]

        print('Parameters to run :', len(parameters))

        # Log parameters to run
        for value in parameters:
            string = (stock + '_candle_' + str(value[0]) + '_strike_' + str(value[2]) +
                      '_entry_' + str(value[3]).replace(':', ',') +
                      '_exit_' + '_' + start_date + '_' + end_date)
            print(string)
        
        # Begin multiprocessing-based execution
        start_time = time.time()
        num_processes = 12
        print('No. of processes :', num_processes)

        # Create a partial function for multiprocessing
        partial_process = partial(parameter_process, mapped_days=mapped_days, option_data=option_data,
                                  df=df, start_date=start_date, end_date=end_date,
                                  counter=counter, output_folder_path=output_folder_path)

        # Start multiprocessing pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(parameters), desc='Processing', unit='Iteration') as pbar:
                def update_progress(combinations):
                    with open(txt_file_path, 'a') as fp:
                        fp.write(str(combinations) + '\n')   # Log completed parameter
                    pbar.update()

                arg_tuples = [tuple(parameter) for parameter in parameters]  # Convert list to tuples for multiprocessing
                
                for result in pool.imap_unordered(partial_process, arg_tuples):
                    update_progress(result)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Time taken to get Initial Tradesheets:', elapsed_time)

# Final print once all processing is complete
print('Finished at :', time.time())
