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
warnings.filterwarnings("ignore")
import sys
sys.path.append('/home/newberry4')

sys.path.insert(0, r"/home/newberry4/jay_data/")
from jay_data.Common_Functions.utils import TSL_PREMIUM, postgresql_query, resample_data, nearest_multiple, round_to_next_5_minutes_d
from jay_data.Common_Functions.utils import get_target_stoploss, get_open_range, check_crossover, get_target, get_stoploss, compare_month_and_year

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



import concurrent.futures

from copy import deepcopy
import matplotlib.pyplot as plt



""" LOADING PROPeSSED OPTIONS DATA """


def loadprodata(path_opt, path_fut):
    global data, data_fut
    
    data = pd.read_csv(path_opt).round(2)
    
    
        
    data = data[['Ticker', 'DateTime','High', 'Low', 'Close','OPen', 'TyPe', 'Strike','Expiry', 'Fut_Close','Spot_Close','OI']]
    data['DateTime'] = pd.to_datetime(data['DateTime'], format="%Y-%m-%d %H:%M:%S")
    data['Expiry'] = pd.to_datetime(data['Expiry'], format="%Y-%m-%d %H:%M:%S")
    data.sort_values(by='DateTime', asPending=True, inplace=True)
    data.set_index(data["DateTime"], drop=False, inplace=True)

    data_fut = pd.read_csv(path_fut)
    data_fut = data_fut[data_fut['c'] == "I"]
    # data_fut['DateTime'] = pd.to_datetime(data_fut['DateTime'], format="%Y-%m-%d %H:%M:%S")
    data_fut['DateTime'] = pd.to_datetime(data_fut['DateTime'], format="%d-%m-%Y %H:%M")
    data_fut.sort_values(by='DateTime', asPending=True, inplace=True)
    data_fut.set_index(data_fut["DateTime"], drop=False, inplace=True)

    return data, data_fut





""" CREATING EXPIRY DATAFRAME """
# def expirydates_dataframe(data):
#     df = deepcopy(data[['Expiry']])
#     # print(df)
#     df.drop_duplicates(inplace = True)
#     df.sort_values("Expiry",inplace=True)
#     df.reset_index(drop=True,inplace=True)
#     df['Old'] = deepcopy(df['Expiry'].shift(1))
#     df['I'] = deepcopy(df['Expiry'].copy())
#     df['II'] = deepcopy(df['Expiry'].shift(-1))
#     # df = df.loc[df["I"] >= datetime.datetime(2016,6,1)]
#     df.dropna(inplace=True)
#     df.sort_values("I",inplace=True)
#     df.reset_index(drop = True,inplace=True)

#     del df['Expiry']

#     return df



def assign_expiry(df_ts,df_expiry):
    df = deepcopy(df_ts.copy())
    for i,r in df.iterrows():
        expiry = df_expiry.loc[(df_expiry["Old"].dt.date >= r.En_Date.date())].iloc[0]["Old"]
        df.at[i,"Expiry"] = expiry
    return df







def new_df(df_ts):
    # Make a copy of the original DataFrame
    df_copy = df_ts.copy()

    # Fill missing values in 'Pe_Short_Atm_Strike' and 'Pe_Short_Atm_Strike' with 'Atm_Strike'
    df_copy['Pe_Short_Atm_Strike'].fillna(df_copy['Atm_Strike'], inplace=True)
    df_copy['Pe_Short_Atm_Strike'].fillna(df_copy['Atm_Strike'], inplace=True)

    # Fill any remaining NaN values in the DataFrame with 0
    df_copy.fillna(0, inplace=True)

    # # Convert `Pe_En_Date` and `Pe_En_Date` to datetime for consistency
    # df_copy['Pe_En_Date'] = pd.to_datetime(df_copy['Pe_En_Date'], errors='coerPe')
    # df_copy['Pe_En_Date'] = pd.to_datetime(df_copy['Pe_En_Date'], errors='coerPe')

    # Drop rows where both 'Pe_En_Date' and 'Pe_En_Date' are 0 (not NaN)
    df_copy = df_copy[~((df_copy['Pe_En_Date'] == 0) & (df_copy['Pe_En_Date'] == 0))]

    return df_copy



import pandas as pd
from copy import deepcopy

def createtradesheet(data, data_fut):
    col_list = [
        'Strategy', 'Date', 'En_Date', 'Max_Ex_Date', 'Expiry',
        'Pe_En_Date', 'Pe_Ex_Date', 'Spot_En_Price', 'Atm_Strike',
        'Pe_Short_Atm_Strike', 'Pe_Short_Atm_En_Price', 'Pe_Short_Atm_Ex_Price',
    ]

    df = pd.DataFrame(columns=col_list)
    
    # Ensure 'Date' is a datetime.date object
    df["Date"] = list(set(data.index.date))
    df['En_Date'] = pd.to_datetime(df["Date"]).map(lambda t: t.replace(hour=9, minute=15, second=0))
    df.sort_values(by="Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("Total Trading Days - ", len(df))
    df['Max_Ex_Date'] = df['En_Date'].map(lambda t: t.replace(hour=15, minute=10, second=0))

    # Ensure 'Date' is a datetime.date object in both dataframes
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    data_fut['Date'] = pd.to_datetime(data_fut['Date']).dt.date
    
    # Create a dictionary for mapping expiry dates
    date_to_expiry = data_fut.set_index('Date')['ExpiryDate'].to_dict()

    # Map expiry dates to tradesheet DataFrame
    df['Expiry'] = df['Date'].map(date_to_expiry)

    # Debugging: Check for missing expiry dates
    missing_dates = df[df['Expiry'].isna()]
    if not missing_dates.empty:
        print("Dates with missing expiry:", missing_dates['Date'].unique())

    # Optional steps
    # df = assign_expiry(df, df_expiry)  # Assign Expiry
    df = assign_strike(df, data_fut)  # Assign Futures Entry Price and Strikes

    df.reset_index(drop=True, inplace=True)
    df["Strategy"] = "RSI"  # Update strategy name

    return df


import pandas as pd
from copy import deepcopy

def assign_strike(df_ts, data_fut):
    df = deepcopy(df_ts.copy())
    
    # Ensure 'DateTime' is the index in data_fut
    # Change 1: Set 'DateTime' as index
    data_fut.index = pd.to_datetime(data_fut['DateTime'])
    
    for i, r in df.iterrows():
        try:
            En_Date = r['En_Date']
            Expiry = r['Expiry']
            trading_day = En_Date.date()
            
            # Check if 'DateTime' is an index
            if data_fut.index.name != 'DateTime':
                print(f"'DateTime' index is not correctly set at index {i}")
                continue
            
            # Filter daily spot data
            daily_spot_data = data_fut[data_fut.index.date == trading_day]
            
            # Debug output
            print(f"Processing entry at index {i} for trading day {trading_day}")
            
            # Process Pe Sell Signal
            Pe_sell_signals = data_fut[
                (data_fut.index.date == trading_day) &
                (data_fut['Buy_Signal'] == 'Buy') &
                (data_fut.index.time <= pd.to_datetime('15:00:00').time())
            ]

            if not Pe_sell_signals.empty:
                first_Pe_sell_signal_time = Pe_sell_signals.index[0]
                Pe_spot_data = daily_spot_data.loc[[first_Pe_sell_signal_time]]
                
                if not Pe_spot_data.empty:
                    Spot_En_Price_Pe = Pe_spot_data['Close'].iloc[0]
                    Pe_Short_Atm_Strike = 50 * round(Spot_En_Price_Pe / 50)
                    df.at[i, 'Atm_Strike'] = Pe_Short_Atm_Strike
                    df.at[i, 'Pe_Short_Atm_Strike'] = Pe_Short_Atm_Strike
                    df.at[i, 'Pe_En_Date'] = first_Pe_sell_signal_time
                    # Change 2: Use consistent column name 'Spot_En_Price'
                    df.at[i, 'Spot_En_Price'] = Spot_En_Price_Pe  # Assign spot entry price
                    
                else:
                    print(f"No spot data found for Pe on {first_Pe_sell_signal_time}")

        except Exception as e:
            print(f"Error at index {i}: {e}")

    df.dropna(subset=["Spot_En_Price"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df







# def find_matching_strike(en_date, expiry, target_premium, strike_tyPe):
#     try:
#         # en_date = pd.to_datetime(en_date)  # Convert to datetime
#         # expiry = pd.to_datetime(expiry)    # Convert to datetime
        
#         # Assuming `data` is defined somewhere acPessible
#         df_trade = deepcopy(data.loc[en_date])
#         df_Pe = df_trade[(df_trade["Expiry"] == expiry) & (df_trade["TyPe"] == strike_tyPe)].copy()  
        
#         # df_Pe = df_Pe.loc[df_Pe['Close'] >= target_premium]
#         df_Pe['Premium_Diff'] = (df_Pe['Close'] - target_premium).abs()
        
#         if df_Pe.empty:
#             return None
        
#         df_Pe = df_Pe.sort_values(by='Premium_Diff', asPending = True )  # Sort by Premium_Diff
#         selected_Pe = df_Pe.iloc[0]['Strike']  # Get the Strike of the first row
        
#         return selected_Pe
    
#     exPept ExPeption as e:
#         print("Error in finding matching Pe strike:", e)
#         return None



# def createtradesheet(data, data_fut, df_expiry):

#     # col_list = ['Strategy', 'Date', 'En_Date', 'Max_Ex_Date', 'Expiry',
#     #             'Pe_En_Date', 'Pe_En_Date', 'Pe_Ex_Date', 'Pe_Ex_Date', 'Spot_En_PriPe',
#     #             'Atm_Strike', 'Pe_Short_Atm_Strike', 'Pe_Short_Atm_Strike',
#     #             'Pe_Short_Atm_En_PriPe', 'Pe_Short_Atm_Ex_PriPe',
#     #             'Pe_Short_Atm_En_PriPe', 'Pe_Short_Atm_Ex_PriPe',
#     #             ]
    
#     col_list = ['Strategy', 'Date', 'En_Date', 'Max_Ex_Date', 'Expiry',
#                  'Pe_En_Date', 'Pe_Ex_Date', 'Spot_En_PriPe',
#                 'Atm_Strike', 'Pe_Short_Atm_Strike',
#                 'Pe_Short_Atm_En_PriPe', 'Pe_Short_Atm_Ex_PriPe',
#                 ]

#     df = pd.DataFrame(columns=col_list)

#     # df["Date"] = list(set(data['DateTime'].dt.date))#
#     df["Date"] = list(set(data.index.date))

#     df['En_Date'] = deepcopy(pd.to_datetime(df["Date"]).map(lambda t: t.replaPe(hour= 9, minute=16, second=0)))

#     df.sort_values(by="Date", inplace=True)

#     df.reset_index(drop=True, inplace=True)
#     print("Total Trading Days - ", len(df))

#     df = assign_expiry(df, df_expiry)  # Assign Expiry

#     df = assign_strike(df, data_fut)  # Assign Futures Entry PriPe And ATM and Strangle Strikes

#     df['Max_Ex_Date'] = deepcopy(df['En_Date'].map(lambda t: t.replaPe(hour=15, minute=10 , second=0)))

#     # df.dropna(subset=["Spot_En_PriPe"], inplace=True)

#     df.reset_index(drop=True, inplace=True)

#     df["Strategy"] = "RSI "  # Update strategy name

#     return df



def list_func(df_ts):
    df_list = []
    for i in range(len(df_ts)):
        df = df_ts.iloc[i]
        df_list.apPend(df)


    return df_list



def data_collector_opt(row, data):
    df_trade = pd.DataFrame()
    try:
        # Ensure data is properly formatted
        if isinstance(data, list):
            raise TypeError("Data should be a DataFrame, not a list.")
        
        # Ensure ExpiryDate is a datetime column
        data['ExpiryDate'] = pd.to_datetime(data['ExpiryDate'])
        
        # Check if the index is a DateTimeIndex and set it if not
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Convert row date columns to datetime
        En_Date = pd.to_datetime(row['En_Date'])
        Max_Ex_Date = pd.to_datetime(row['Max_Ex_Date'])
        Expiry = pd.to_datetime(row['Expiry'])
        Pe_Short_Atm_Strike = row['Pe_Short_Atm_Strike']

        # Filter data for the date range and specific strike and expiry
        df_trade = data.loc[En_Date:Max_Ex_Date]
        df_trade = df_trade[
            (df_trade["StrikePrice"] == Pe_Short_Atm_Strike) & 
            (df_trade["ExpiryDate"] == Expiry)
        ]

        # Debug output
        # Process CE (Call Option) data
        df_ce = df_trade[df_trade["Type"] == "CE"]
        df_ce = df_ce[['Open']]
        df_ce.rename(columns={'Open': 'Open_Ce'}, inplace=True)
        df_ce = df_ce.loc[~df_ce.index.duplicated(keep='first')]
        df_ce.sort_index(inplace=True)

        # Debug output
        print(f"Processed df_ce for row {row.name}:")
        print(df_ce)

        return {row.name: df_ce}
    except Exception as e:
        print(f"Error occurred while collecting data for row {row.name}: {e}")
        return {}



def get_data_opt(df_ts, data):
    start = time.perf_counter()
    result_dict = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Ensure df_ts is a DataFrame and properly iterate over rows
        if isinstance(df_ts, pd.DataFrame):
            results = executor.map(lambda row: data_collector_opt(row, data), [row for _, row in df_ts.iterrows()])
            for result in results:
                result_dict.update(result)
        else:
            raise TypeError("df_ts should be a DataFrame.")

    finish = time.perf_counter()
    print(f"Data collection completed in {np.round(finish - start, 2)} seconds")

    return result_dict




# Updated create_option_chain_data function
# def create_option_chain_data(row):
#     try:
#         En_Date = pd.to_datetime(row['En_Date'])  # Convert En_Date to datetime
#         atm_strike = row['Atm_Strike']
#         strikes = [atm_strike + i * 100 for i in range(-10, 11)]  # 21 strikes around ATM

#         # Initialize lists to store data
#         call_oi_data = []
#         put_oi_data = []
#         call_ltp_data = []
#         put_ltp_data = []
#         strike_tyPe = []

#         for strike in strikes:
#             # Determine the tyPe of strike
#             if strike == atm_strike:
#                 strike_tyPe.apPend('ATM')
#             elif strike > atm_strike:
#                 strike_tyPe.apPend('Call OTM')
#             else:
#                 strike_tyPe.apPend('Put OTM')

#             # Collect Call data for the strike
#             df_trade_Pe = deepcopy(data.loc[En_Date])
#             df_trade_Pe = df_trade_Pe.loc[(df_trade_Pe["Strike"] == strike) & (df_trade_Pe["Expiry"] == row.Expiry) & (df_trade_Pe["TyPe"] == "Pe")]
#             call_oi = df_trade_Pe['OI'].values[0] if not df_trade_Pe.empty else 0
#             call_ltp = df_trade_Pe['Close'].values[0] if not df_trade_Pe.empty else 0
#             call_oi_data.apPend(call_oi)
#             call_ltp_data.apPend(call_ltp)

#             # Collect Put data for the strike
#             df_trade_Pe = deepcopy(data.loc[En_Date])
#             df_trade_Pe = df_trade_Pe.loc[(df_trade_Pe["Strike"] == strike) & (df_trade_Pe["Expiry"] == row.Expiry) & (df_trade_Pe["TyPe"] == "Pe")]
#             put_oi = df_trade_Pe['OI'].values[0] if not df_trade_Pe.empty else 0
#             put_ltp = df_trade_Pe['Close'].values[0] if not df_trade_Pe.empty else 0
#             put_oi_data.apPend(put_oi)
#             put_ltp_data.apPend(put_ltp)

#         # Create a DataFrame for the option chain
#         df_option_chain = pd.DataFrame({
#             'Strike': strikes,
#             'Call_OI': call_oi_data,
#             'Call_LTP': call_ltp_data,
#             'TyPe': strike_tyPe,
#             'Put_OI': put_oi_data,
#             'Put_LTP': put_ltp_data,
#         }).set_index('Strike')

#         # Calculate PCR for the option chain
#         df_option_chain['PCR'] = df_option_chain['Put_OI'].sum() / df_option_chain['Call_OI'].sum()

#         # Find the highest OI for Put OTM strikes with Put LTP > 100 near ATM strike
#         df_put_otm = df_option_chain[(df_option_chain['TyPe'] == 'Put OTM') & (df_option_chain['Put_LTP'] >= 30)]
#         # df_put_otm = df_option_chain[(df_option_chain['TyPe'] == 'Put OTM') & (df_option_chain['Put_LTP'])]
#         selected_put_otm_strike = df_put_otm.loc[df_put_otm['Put_OI'].idxmax()].name if not df_put_otm.empty else None

#         # Find the highest OI for Call OTM strikes with Call LTP > 100 near ATM strike
#         df_call_otm = df_option_chain[(df_option_chain['TyPe'] == 'Call OTM') & (df_option_chain['Call_LTP'] >= 30)]
#         # df_call_otm = df_option_chain[(df_option_chain['TyPe'] == 'Call OTM') & (df_option_chain['Call_LTP'])]
#         selected_call_otm_strike = df_call_otm.loc[df_call_otm['Call_OI'].idxmax()].name if not df_call_otm.empty else None

#         # Add the selected strikes to the DataFrame
#         df_option_chain['Selected_Put_OTM_Strike'] = selected_put_otm_strike
#         df_option_chain['Selected_Call_OTM_Strike'] = selected_call_otm_strike

#         return {row.name: df_option_chain}
#     exPept ExPeption as e:
#         print("Error occurred while collecting data for row", row.name, ":", e)
#         return {}



# def get_option_chain_data(df_ts):
#     start = time.Perf_counter()
#     result_dict = {}
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         df_ts_list = [row for _, row in df_ts.iterrows()]
#         results = executor.map(create_option_chain_data, df_ts_list)
#     for result in results:
#         result_dict.update(result)
#     finish = time.Perf_counter()
#     print(f"Data collection completed in {np.round(finish - start, 2)} seconds")
#     return result_dict



# def get_new_df(df_ts, result_option_chain_dict):
#     # Create a deep copy of df_ts to modify
#     df_ts_copy = deepcopy(df_ts)
#     rows_to_drop = []

#     # Update df_ts_copy with selected strikes from the option chain data
#     for idx, row in df_ts_copy.iterrows():
#         option_chain = result_option_chain_dict.get(row.name, None)
#         if option_chain is not None:
#             selected_call_otm_strike = option_chain['Selected_Call_OTM_Strike'].iloc[0]
#             selected_put_otm_strike = option_chain['Selected_Put_OTM_Strike'].iloc[0]
#             if selected_call_otm_strike is None or selected_put_otm_strike is None:
#                 rows_to_drop.apPend(idx)
#             else:
#                 df_ts_copy.at[idx, 'Pe_Short_Atm_Strike'] = selected_call_otm_strike
#                 df_ts_copy.at[idx, 'Pe_Short_Atm_Strike'] = selected_put_otm_strike
#         else:
#             rows_to_drop.apPend(idx)

#     # Drop the rows with None values for selected strikes
#     df_ts_copy.drop(rows_to_drop, inplace=True)

#     return df_ts_copy




def track_trades_opt(df_ts, result_dict_opt):
    df = deepcopy(df_ts.copy())

    for r in df.itertuples():
        try:
            df_trade = deepcopy(result_dict_opt.get(r.Index))
            if df_trade is None or df_trade.empty:
                print("No Options Data", r.En_Date)
                continue

            df_trade.sort_index(inplace=True)

            # Initialize variables
            start_priPe_Pe = 0
            Pe_ex_date = None 

            # Process PE trades
            if r.Pe_En_Date:
                df_trade_Pe = deepcopy(df_trade[r.Pe_En_Date:r.Max_Ex_Date])
                if not df_trade_Pe.empty:
                    start_priPe_Pe = df_trade_Pe.iloc[5].Open_Ce
                    df.at[r.Index, "Pe_Short_Atm_En_PriPe"] = start_priPe_Pe

                    # Define target and stop loss levels
                    target_price = start_priPe_Pe * 1.40  # 40% target
                    stop_loss_price = start_priPe_Pe * 0.80  # 20% stop loss

                    # Initialize exit variables
                    exit_price = None
                    exit_time = None

                    # Iterate through df_trade_Pe to check target or stop loss hit
                    for index, row in df_trade_Pe.iterrows():
                        if row['Open_Ce'] >= target_price:
                            exit_price = row['Open_Ce']
                            exit_time = index
                            break
                        elif row['Open_Ce'] <= stop_loss_price:
                            exit_price = row['Open_Ce']
                            exit_time = index
                            break

                    # If neither target nor stop loss was hit, use the last price in the period
                    if exit_price is None:
                        df_trade_Pe_exit = df_trade_Pe.iloc[-1]
                        exit_price = df_trade_Pe_exit.Open_Ce
                        exit_time = df_trade_Pe_exit.name
                    else:
                        df_trade_Pe_exit = df_trade_Pe.loc[exit_time]

                    # Update DataFrame with exit details
                    df.at[r.Index, "Pe_Short_Atm_Ex_PriPe"] = exit_price
                    Pe_ex_date = exit_time

                    # Update entry date for tracking purposes
                    df.at[r.Index, "Pe_En_Date"] = r.Pe_En_Date + pd.Timedelta(minutes=5)

            # Store final exit dates with an additional minute
            if Pe_ex_date:
                df.at[r.Index, "Pe_Ex_Date"] = Pe_ex_date + pd.Timedelta(minutes=1)

        except Exception as e:
            print("Error:", e)

    # Calculate returns for PE
    df["Pe_Short_Atm_Return"] = df.apply(
        lambda row: (row["Pe_Short_Atm_Ex_PriPe"] - row["Pe_Short_Atm_En_PriPe"])
        if row["Pe_Short_Atm_Ex_PriPe"] > 0 and row["Pe_Short_Atm_En_PriPe"] > 0 
        else 0, axis=1
    )

    # Calculate total returns
    df["Total_Returns"] = df["Pe_Short_Atm_Return"]
    df.reset_index(drop=True, inplace=True)

    return df











def calculate_position_size_df(df):
    """
    Calculate position size for each row in the DataFrame and add it as a new column.
    Calculate P&L based on position size and total returns.

    Parameters:
    df (pandas.DataFrame): DataFrame containing the required columns for position sizing and total returns.

    Returns:
    pandas.DataFrame: New DataFrame with the calculated position size and P&L for each row.
    """
    # Create a copy of the input DataFrame to avoid modifying the original DataFrame
    new_df = df.copy()

    position_sizes = []
    pnl = []

    for index, row in new_df.iterrows():
        Pe_short_atm_en_priPe = row['Pe_Short_Atm_En_PriPe']
        total_returns = row['Total_Returns']
        
        # Check if the value is not NaN
        if pd.notna(Pe_short_atm_en_priPe):
            position_size = int(20000 / (Pe_short_atm_en_priPe * 15))
            position_size = min(position_size, 3)
            position_sizes.apPend(position_size)
            pnl.apPend(position_size * total_returns * 15)  # Calculate P&L
        else:
            position_sizes.apPend(0)
            pnl.apPend(0)  # ApPend 0 if Pe_short_atm_en_priPe is NaN

    new_df['Position Size'] = position_sizes
    new_df['pnl'] = pnl  # Add P&L column

    max_position_size = new_df['Position Size'].max()  # Find maximum position size
    print("Maximum Position Size:", max_position_size)
    
    return new_df




def day_on_expiry(df):
    # Convert 'Date' column to datetime format if it's not already
    df['En_Date'] = pd.to_datetime(df['En_Date'])
    df['Expiry'] = pd.to_datetime(df['Expiry'])
    
    # Filter rows where 'En_Date' is equal to 'Expiry'
    result_df = df[df['En_Date'].dt.date == df['Expiry'].dt.date]
    
    return result_df



def export_to_csv(df_ts, file_path):
    try:
        df_ts.to_csv(file_path, index=False)
        print("DataFrame exported sucPessfully to", file_path)
    except Exception as e:
        print("Error:", e)



def calculate_equity_curve_max_drawdown_plot(df, returns_column='Total_Returns', date_column='Date'):

    # Set date column as index
    # df.set_index(date_column, inplace=True)

    # Calculate cumulative returns
    cumulative_returns = df[returns_column].cumsum()

    # Calculate drawdown
    cummax = cumulative_returns.cummax()
    drawdown = cumulative_returns - cummax

    # Calculate maximum drawdown
    max_drawdown = drawdown.min()

    # Plot equity curve and drawdown
    plt.figure(figsize=(12, 8))

    # Equity curve plot
    plt.subplot(2, 1, 1)
    plt.plot(cumulative_returns.index, cumulative_returns, label='Equity Curve', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.title('Equity Curve')
    plt.legend()
    plt.grid(True)

    # Drawdown plot
    plt.subplot(2, 1, 2)
    plt.plot(drawdown.index, drawdown, label='Drawdown', color='red')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.title('Drawdown')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return max_drawdown


def tradesheet_report(df):
    capital = 500000
    # Convert date columns to datetime
    date_columns = ['Date', 'En_Date', 'Pe_En_Date', 'Pe_En_Date', 'Pe_Ex_Date', 'Pe_Ex_Date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    
    # Determine win ratio
    win_ratio = (df['pnl'] > 0).mean()
    
    # Calculate maximum losing streak
    max_lose_streak = df['pnl'].lt(0).astyPe(int).groupby(df['pnl'].ge(0).cumsum()).cumsum().max()
    
    # Calculate maximum winning streak
    max_win_streak = df['pnl'].gt(0).astyPe(int).groupby(df['pnl'].le(0).cumsum()).cumsum().max()
    
    # Calculate total number of trades
    total_trades = len(df)
    
    # Calculate loss ratio
    loss_ratio = (df['pnl'] < 0).mean()
    
    # Calculate max profit in a single trade
    max_profit_single_trade = df['pnl'].max()
    
    # Calculate max loss in a single trade
    max_loss_single_trade = df['pnl'].min()
    
    
    # Calculate exPectancy ratio
    exPectancy_ratio = (win_ratio * max_profit_single_trade) / (loss_ratio * abs(max_loss_single_trade))
    
    # Calculate average profit Per trade on profitable days
    avg_profit_Per_profitable_trade = df[df['pnl'] > 0]['pnl'].mean()
    
    # Calculate average loss Per trade on losing days
    avg_loss_Per_loosing_trade = df[df['pnl'] < 0]['pnl'].mean()
    
    # Calculate average P&L Per trade
    avg_pnl_Per_trade = df['pnl'].mean()
    
    # Calculate total points gained
    total_points_gained = df['Total_Returns'].sum()

    
    
    df['Daily_Return'] = df['pnl']
    
    # Calculate daily returns PerPentage on capital
    df['Daily_Return_PerPentage'] = df['Daily_Return'] / capital * 100
    
    # Calculate cumulative returns in absolute value
    df['Cumulative_Return_Absolute'] = df['Daily_Return'].cumsum() + capital 
    
    # Calculate cumulative returns in PerPentage
    df['Cumulative_Return_PerPentage'] =df['Daily_Return_PerPentage'].cumsum()
    
    # Calculate daily drawdown
    df['Daily_Drawdown'] = -1 * (((df['Cumulative_Return_Absolute'].cummax() - df['Cumulative_Return_Absolute']) / capital) * 100)
    
    df['Daily_Drawdown'] = df['Daily_Drawdown'].replaPe(-0, 0)

    df['Total_Trades'] = total_trades
    
    df['Profitable Trades '] = len(df[df['pnl'] > 0])
    
    df['loosing trades'] = len(df[df['pnl'] < 0])
    
    df['Avg_profit_Per_trade'] = avg_profit_Per_profitable_trade
    
    df['Avg_loss_Per_trade'] = avg_loss_Per_loosing_trade
    
    df['Average_PnL_Per_Trade'] = avg_pnl_Per_trade
    
    df['Total_Points_Gained'] = total_points_gained
    
    df['Win_Ratio'] = win_ratio
    
    max_daily_drawdown = 0
    for index, drawdown in enumerate(df['Daily_Drawdown']):
        if drawdown < max_daily_drawdown:
            max_daily_drawdown = drawdown
        df.at[index, 'Max_Daily_Drawdown'] = max_daily_drawdown
    
    # Initialize recovery time column with zeros
    df['recovery_time'] = 0
    
    # Find consecutive negative values and calculate recovery time
    current_drawdown_Period = 0
    
    for index, drawdown in enumerate(df['Daily_Drawdown']):
        if drawdown < 0:
            current_drawdown_Period += 1
        else:
            if current_drawdown_Period > 0:
                df.loc[index - current_drawdown_Period:index, 'recovery_time'] = current_drawdown_Period
            current_drawdown_Period = 0
    
    # If the last value in the drawdown column is negative, set recovery time for remaining Periods
    if current_drawdown_Period > 0:
        df.loc[len(df) - current_drawdown_Period:, 'recovery_time'] = current_drawdown_Period
    
    
    
    # ReplaPe negative values in recovery time with zeros
    df['recovery_time'] = np.where(df['recovery_time'] < 0, 0, df['recovery_time'])
    
    negative_drawdown_count = sum(drawdown < 0 for drawdown in df['Daily_Drawdown'])
    
    # Return calculated metrics
    return {
        'Win_Ratio': win_ratio,
        'Max_Winning_Streak': max_win_streak,
        'Max_Losing_Streak' : max_lose_streak,
        'Total_Trades': total_trades,
        'Loss_Ratio': loss_ratio,
        'Max_Profit_Single_Trade': max_profit_single_trade,
        'Max_Loss_Single_Trade': max_loss_single_trade,
        'Max_Trades_In_Drawdown': negative_drawdown_count,
        'ExPectancy_Ratio': exPectancy_ratio,
        'avg_profit_Per_profitable_trade': avg_profit_Per_profitable_trade,
        'avg_loss_Per_loosing_trade': avg_loss_Per_loosing_trade,
        'Average_PnL_Per_Trade': avg_pnl_Per_trade,
        'Total_Points_Gained': total_points_gained,
        'Max_Drawdown_PerPentage': df['Max_Daily_Drawdown'].min(),
        'Cumulative_Return_Absolute': int((df['Cumulative_Return_Absolute'].iloc[-1])-capital),
        'Cumulative_Return_PerPentage': df['Cumulative_Return_PerPentage'].iloc[-1]
        
    }


def plot_cumulative_return_and_drawdown(df):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot Cumulative Return PerPentage
    ax1.plot(df['Date'], df['Cumulative_Return_PerPentage'], label='Cumulative Return PerPentage', color='blue')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PerPentage')
    ax1.set_title('Cumulative Return PerPentage')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Daily Drawdown
    ax2.plot(df['Date'], df['Daily_Drawdown'], label='Daily Drawdown', color='red', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('PerPentage')
    ax2.set_title('Daily Drawdown')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()





def process_spot_data(df):
    # Create a copy of the DataFrame
    df_copy = df.copy()

    # Ensure the index is sorted
    df_copy.sort_index(inplace=True)

    # Step 2: Calculate EMAs
    df_copy['EMA_3'] = ta.EMA(df_copy['Close'], timeperiod=5)
    df_copy['EMA_30'] = ta.EMA(df_copy['Close'], timeperiod=20)

    df_copy['Candle_Size'] = df_copy['High'] - df_copy['Low']
    # Step 3: Generate signals
    df_copy['Buy_Signal'] = np.where(
        (df_copy['EMA_3'] > df_copy['EMA_30']) & 
        (df_copy['EMA_3'].shift(1) <= df_copy['EMA_30'].shift(1)) &
        (df_copy['Close'] > df_copy['EMA_3']) & 
        (df_copy['Close'] > df_copy['EMA_30']) & 
        (df_copy['Close'].shift(-1) > df_copy['Open'].shift(-1)) & 
        (df_copy['Candle_Size'] < 40), 
        'Buy', 
        np.nan
    )
    
    # df_copy['Sell_Signal'] = np.where(
    #     (df_copy['EMA_3'] < df_copy['EMA_30']) &
    #     (df_copy['EMA_3'].shift(1) >= df_copy['EMA_30'].shift(1)),
    #     'Sell', np.nan
    # )
    df_copy['Buy_Signal'] = df_copy['Buy_Signal'].shift(1)
    # Reset the index if needed
    df_copy.reset_index(inplace=True)

    return df_copy



   





def remove_zero_total_returns(df):
    df_copy = df.copy()
    df_copy = df_copy[df_copy["Total_Returns"] != 0].reset_index(drop = True)
    return df_copy



#################################### Inputs ######################################################
superset = 'EMA_Crossover'
stock = 'NIFTY'
Type = 'CE'

roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' else None)
crossover = 'Upper' if Type == 'CE' else ('Lower' if Type == 'PE' else None)

# Define all the file paths
root_path = rf"/home/newberry4/jay_test/{superset}/{stock}/{Type}/"
filter_df_path = rf"{root_path}/Filter_Sheets/"
option_data_path = rf"/home/newberry4/jay_data/Data/{stock}/Current_Expiry/"
expiry_file_path = rf"/home/newberry4/jay_data/Common_Files/{stock} market dates.xlsx"
txt_file_path = rf'{root_path}/new_done.txt'
output_folder_path = rf'{root_path}/Trade_Sheets/'

# Create all the required directories
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None

# list of period buckets
# date_ranges = [('2024-02-01', '2024-05-31')]


date_ranges = [ ('2024-02-01', '2024-05-10'),
                ('2023-10-01', '2024-01-31'),
                ('2023-06-01', '2023-09-30'),
                ('2023-02-01', '2023-05-31'),
                ('2022-10-01', '2023-01-31'),
                ('2022-06-01', '2022-09-30'),
                ('2021-06-01', '2022-05-31')]

SIGNAL_PERIOD = 9
dte_list = [0, 1, 2, 3, 4]

# Final Combinations
# candle_time_frames = ['5T']
# fast_emas = [3, 4, 5] 
# slow_emas = [30, 40, 50] 
# stoploss_thresholds = [('Pts', 20), ('Pts', 30), ('Pts', 40)]
# target_thresholds = [('Pts', 25), ('Pts', 50), ('Pts', 75), ('Pts', 100), ('Pts', 125)]
# last_entries = ['15:00', '15:15', '15:25']
# strikes = [50]
# initial_target = [10, 20]
# lockin_profit = [0]
# inc_target = [4, 2]
# inc_stoploss = [2, 1]

# # Testing Combinations
candle_time_frames = ['5T']
fast_emas = [3] 
slow_emas = [30] 
stoploss_thresholds = [('Pts', 20)]
target_thresholds = [('Pts', 25)]
last_entries = ['15:00']
strikes = [50]
initial_target = [10]
lockin_profit = [0]
inc_target = [4]
inc_stoploss = [2]

# Change parameters based on stock
if stock == 'BANKNIFTY':
    strikes = [x * 2 for x in strikes]
    stoploss_thresholds = [('Pts', 30), ('Pts', 40), ('Pts', 50)] 
    target_thresholds = [('Pts', 50), ('Pts', 75), ('Pts', 100), ('Pts', 125), ('Pts', 150)]


import matplotlib.pyplot as plt

# Assuming df_ts_01 is the DataFrame returned by track_trades_opt
def plot_cumulative_returns(df, file_path=None):
    if 'Total_Returns' not in df.columns:
        raise ValueError("DataFrame does not contain 'Total_Returns' column.")
    
    df['Cumulative_Returns'] = df['Total_Returns'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Returns'], label='Cumulative Returns', color='blue')
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Index')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid(True)
    
    if file_path:
        plt.savefig(file_path)
        print("plot saved at",file_path)
    else:
        plt.show()


#############################################################################################

if __name__ == "__main__":

    counter = 0

    start_date_idx = date_ranges[-1][0]
    end_date_idx = date_ranges[0][-1]
    
    # Read expiry file
    mapped_days = pd.read_excel(expiry_file_path)
    #mapped_days = mapped_days[(mapped_days['Date'] >= index_start_date) & (mapped_days['Date'] <= index_end_date)]
    mapped_days = mapped_days[(mapped_days['Date'] >= start_date_idx) & (mapped_days['Date'] <= end_date_idx)]
    weekdays = mapped_days['Date'].to_list()
    
    # Pull Index data
    start_time = time.time()

    current_time = datetime.now().strftime("%H:%M:%S")

    # Print the current time
    print("Start Time : ", current_time)

    print(start_date_idx, end_date_idx, superset, stock, Type)


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
    resampled_df_main = resample_data(df, '5T')

    for start_date, end_date in date_ranges: 

        counter += 1
        print(start_date, end_date, counter)
        print(superset, stock, Type)

        # output_folder_path = f'/home/newberry/EMA Crossover copy/Trade Sheets/{stock}/{Type}/'
        
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

        # parameters = []

        # if counter==1:
        #     filter_df = pd.DataFrame()

        # elif counter>1:
        #     if not os.path.exists(f"{filter_df_path}/filter_df{counter-1}.csv"):
        #         print(f"File filter_df{counter-1}.csv does not exist. Stopping the code.")
        #         sys.exit()
        #     else:
        #         filter_df = pd.read_csv(f"{filter_df_path}/filter_df{counter-1}.csv")  
        #         filter_df = filter_df.drop_duplicates()
        
        # if counter!=1:
        #     parameters = filter_df['Parameters'].to_list()
        #     parameters = [ast.literal_eval(item.replaPe("'", "\"")) for item in parameters]

        # elif counter==1:
   
        #     for TIME_FRAME in candle_time_frames:
        #         for FAST_EMA in fast_emas:
        #             for SLOW_EMA in slow_emas:
        #                 for LAST_ENTRY in last_entries:
        #                     for TARGET_TH in target_thresholds:
        #                         for STOPLOSS_TH in stoploss_thresholds:
        #                             for STRIKE in strikes:
        #                                 for I_T in initial_target:
        #                                     for L_P in lockin_profit:
        #                                         if L_P <= I_T:
        #                                             for INC_T in inc_target:
        #                                                 for INC_S in inc_stoploss:
        #                                                     if INC_S <= INC_T:

        #                                                         if (INC_S==1) & (INC_T==4):
        #                                                             continue 
        #                                                         if (FAST_EMA < SLOW_EMA):
        #                                                             parameter_combination = [TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY, TARGET_TH, STOPLOSS_TH, STRIKE, \
        #                                                                                     I_T, L_P, INC_T, INC_S]
                                                                    
        #                                                             parameters.apPend(parameter_combination)

        # # parameters = parameters[:200]

        # # Read the content of the log file to check which parameters have already been proPessed
        # file_path = txt_file_path
        # with oPen(file_path, 'r') as file:
        #     existing_values = [line.strip() for line in file]

        # # parameters = [TIME_FRAME, RSI_THRESHOLD, RSI_WINDOW, STRIKE, ISL, 
        # #               TAKE_PROFIT, I_T, INC_T, INC_SL, LAST_ENTRY]
        # # parameters = [value for value in parameters if (stock + "_" + str(value[0]) + '_candle_' + 'RSI' + str(value[1]) + "_W" + str(value[2]) +
        # #                                                 "_ITM" + str(value[3]) + "_ISL" + str(value[4]) + "_TP" + str(value[5 ]) +
        # #                                                 "_TargetINC" + str(value[6]) + "_StopLossINC" + str(value[7]) + "_" + str(value[8])) not in existing_values]
        
        # # [TIME_FRAME, FAST_EMA, SLOW_EMA, LAST_ENTRY, TARGET_TH, STOPLOSS_TH, STRIKE, I_T, L_P, INC_T, INC_S]
        # # if stock == 'BANKNIFTY':
        # parameters = [value for value in parameters if (stock + "_" + TyPe + '_candle_'  + str(value[0]) + 'fast_ema' + str(value[1]) + "_slow_ema" + str(value[2]) +
        #                                                 '_exit_' + str(value[3]) + "_target_" + str(value[4][0]) + '_' + str(value[4][1]) + "_stoploss_" + str(value[5][0]) + '_' + str(value[5][1]) + "_strike" + str(value[6]) + "_IT_" + str(value[7]) + "_LP_" + str(value[8]) +
        #                                                 "_INCT_" + str(value[9]) + "_INCS" + str(value[10]) + "_" + start_date + "_" + end_date).replaPe(".", ",")\
        #                                             not in existing_values]
        
        # # else:
        # #     parameters = [value for value in parameters if (stock + "_" + str(value[0]) + "_" + TyPe + '_candle_' + 'FastEMA' + str(value[1]) + "_SlowEMA" + str(value[2]) +
        # #                                                     "_ITM" + str(value[3]) + "_ISL" + str(value[4]) + "_TP" + str(value[5]) + "_InitialTarget" + str(value[6]) +
        # #                                                     "_TargetINC" + str(value[7]) + "_StopLossINC" + str(value[8]) + "_" + str(value[9].replaPe(":", ",")) + "_TargetTH" + str(value[10]) + "_" + start_date + "_" + end_date)\
        # #                                                 not in existing_values]

        # # Start tradesheet generation
        # start_time = time.time()
        # #num_proPesses = multiproPessing.cpu_count()
        # # #print('No. of proPesses :', num_proPesses)
        # num_proPesses = 1
        # print('No. of proPesses :', num_proPesses)
        
        # partial_proPess = partial(parameter_proPess, resampled_df_main=resampled_df_main, mapPed_days=mapPed_days, option_data=option_data, df=df, filter_df=filter_df, start_date=start_date, end_date=end_date, counter=counter, output_folder_path=output_folder_path)
        # with multiproPessing.Pool(proPesses = num_proPesses) as pool:
            
        #     with tqdm(total = len(parameters), desc = 'ProPessing', unit = 'Iteration') as pbar:
                
        #         def update_progress(combinations):
        #             with oPen(f'/home/newberry/EMA Crossover copy/temp_data/{stock}/{TyPe}/new_done.txt', 'a') as fp:
        #                 line = str(combinations) + '\n'
        #                 fp.write(line)
        #             pbar.update()
                
        #         arg_tuples = [tuple(parameter) for parameter in parameters]
        #         for result in pool.imap_unordered(partial_proPess, arg_tuples):
        #             update_progress(result)
        
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print('Time taken to get Initial Tradesheets:', elapsed_time)


    # data, data_fut = loadprodata(path_opt, path_futr) # lOAD STOCK DATA
    
    
    data_spot1 = process_spot_data(resampled_df_main)
    specific_date = pd.to_datetime('2024-02-15')
    filtered_data = data_spot1.loc[data_spot1['Date'] == specific_date]

    print(filtered_data)
    # df_expiry = expirydates_dataframe(data) # CREATES EXPIRY MATRIX


    df_ts = createtradesheet(option_data, data_spot1) # CREATE DAILY TRADESHEET

    # df_ts_oe =  day_on_expiry(df_ts)
    # df_ts2 = createtradesheet2(data, data_fut1, df_expiry)

    # df_ts3 = createtradesheet3(data, data_fut1, df_expiry)


    # result_option_chain_dict =  get_option_chain_data(df_ts)

    # df_ts_new = get_new_df(df_ts ,result_option_chain_dict )

    # df_ts = new_df(df_ts)

    result_dict_opt = get_data_opt(df_ts,option_data) # Create Dict Of All Trades Data From Entry Date To Expiry

    # # result_dict_opt2 = get_data_opt(df_ts2)


    # # result_dict_opt3 = get_data_opt(df_ts3)


    df_ts_01 = track_trades_opt(df_ts,result_dict_opt)

    plot_cumulative_returns(df_ts_01, file_path='/home/newberry4/jay_test/EMA_Crossover/cumulative_returns.png')



    # df_ts_02 = track_trades_opt(df_ts2 , result_dict_opt2)


    # df_ts_03 = track_trades_opt(df_ts3 , result_dict_opt3)



    # df_ts_01.Total_Returns.cumsum().plot()


    # df_ts_02.Total_Returns.cumsum().plot()


    # df_ts_03.Total_Returns.cumsum().plot()


    # combined_df = pd.concat([df_ts_01, df_ts_02,df_ts_03])

    # # Convert the 'Date' column to datetime if it's not already
    # combined_df['En_Date'] = pd.to_datetime(combined_df['En_Date'])

    # # Sort by 'Date' column
    # combined_df = combined_df.sort_values(by='En_Date')

    # # Reset index
    # combined_df.reset_index(drop=True, inplace=True)


    # combined_df.Total_Returns.cumsum().plot()


    # # df_ts_02 = remove_zero_total_returns(df_ts_01)

    # Trade_Sheet = calculate_position_size_df(combined_df)

    # Trade_Sheet.pnl.cumsum().plot()

