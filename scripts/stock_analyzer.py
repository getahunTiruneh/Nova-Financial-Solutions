import pandas as pd
import os
import pynance as pn
import talib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.express as px
import plotly.express as px
import plotly.express as px

class StockPriceAnalyzer:
    def __init__(self, file_path=None, folder_path=None):
        """
        Initializes the StockPriceAnalyzer class with a CSV file path or a folder path.
        
        :param file_path: str (optional) - Path to a single CSV file containing stock data.
        :param folder_path: str (optional) - Path to a folder containing multiple CSV files to be merged.
        """
        self.file_path = file_path
        self.folder_path = folder_path
        self.data = pd.DataFrame()
        
        if self.file_path:
            # Load data from a single CSV file
            self.data = pd.read_csv(file_path)
        elif self.folder_path:
            # Automatically merge CSV files from the specified folder
            self.merge_csv_files(folder_path)

    def merge_csv_files(self, folder_path):
        """
        Merges all CSV files from the specified folder into one DataFrame and adds a 'stock_symbol' column.
        
        :param folder_path: str - Path to the folder containing CSV files.
        :return: DataFrame - Merged DataFrame of all CSV files in the folder.
        """
        merged_df = pd.DataFrame()
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                
                # Extract stock symbol from the file name before the underscore
                stock_symbol = os.path.splitext(file)[0].split('_')[0]  # Extract symbol before the underscore
                
                # Add the stock_symbol column to the DataFrame
                df['stock_symbol'] = stock_symbol
                
                # Merge the DataFrame into the main DataFrame
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        
        self.data = merged_df
        return self.data

    def plot_price(self):
        """
        Plots the close price of the stock over time with respect to each stock symbol using Plotly Express.
        """
        if self.data.empty:
            print("No data to plot. Please load or fetch data first.")
            return
        
        # Ensure that the 'Date' column is in datetime format for better plotting
        self.data['Date'] = pd.to_datetime(self.data['Date'],errors='coerce',utc=True)
        
        # Plot Close price for each stock symbol using Plotly Express
       # Set the figure size
        plt.figure(figsize=(12, 8))

        # Plot Close prices over time for different stock symbols
        for stock_symbol, group_data in self.data.groupby('stock_symbol'):
            plt.plot(group_data['Date'], group_data['Close'], label=stock_symbol)

        # Set title and labels
        plt.title('Close Price Over Time for Different Stock Symbols')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')

        # Rotate and format x-axis labels
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

        # Show legend
        plt.legend(title='Stock Symbol')

        # Display the plot
        plt.show()
        
        #Apply Analysis Indicators with TA-Lib 
    def calculate_technical_indicators(self, data):
        self.data['Data']=pd.to_datetime(self.data['Date'], format='ISO8601')
        # Calculate various technical indicators
        data['SMA'] = talib.SMA(data['Close'], timeperiod=20)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['EMA'] = talib.EMA(data['Close'], timeperiod=20)
        macd, macd_signal, _ = talib.MACD(data['Close'])
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        return data

    def plot_stock_data(self, data, symbol):
        self.data['Date'] = pd.to_datetime(self.data['Date'],errors='coerce',utc=True)
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Close')
        plt.plot(data['Date'], data['SMA'], label='SMA')
        plt.title(f'{symbol} Stock Price with Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_rsi(self, data, symbol):
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['RSI'], label='RSI')
        plt.title(f'{symbol} Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.show()

    def plot_ema(self, data, symbol):
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Close')
        plt.plot(data['Date'], data['EMA'], label='EMA')
        plt.title(f'{symbol} Stock Price with Exponential Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def plot_macd(self, data, symbol):
        plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['MACD'], label='MACD')
        plt.plot(data['Date'], data['MACD_Signal'], label='MACD Signal')
        plt.title(f'{symbol} Moving Average Convergence Divergence (MACD)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    def visualize_stocks(self, stock_data):
        # Loop through each stock symbol
        self.data['Date'] = pd.to_datetime(self.data['Date'],errors='coerce',utc=True)
        for symbol in stock_data['stock_symbol'].unique():
            data = stock_data[stock_data['stock_symbol'] == symbol].copy()
            data = self.calculate_technical_indicators(data)
            
            # Plot all indicators for each stock symbol
            self.plot_stock_data(data, symbol)
            self.plot_rsi(data, symbol)
            self.plot_ema(data, symbol)
            self.plot_macd(data, symbol)
            
            
        # Calculate additional financial metrics using PyNance
    def calculate_financial_metrics(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'],format='ISO8601')
    
        # Calculate daily returns using pandas
        self.data['Daily Returns'] = self.data['Close'].pct_change()

        # Calculate rolling volatility using pandas (standard deviation)
        self.data['Rolling Volatility'] = self.data['Close'].rolling(window=20).std()

        # Use pynance to calculate Simple Moving Average (SMA)
        # Assuming the 'pynance' library provides SMA or other financial metrics
        self.data['Moving Average'] = self.data['Close'].rolling(window=10).mean()

        # Calculate cumulative returns using pandas
        self.data['Cumulative Returns'] = (1 + self.data['Daily Returns']).cumprod() - 1

        return self.data
    
        #Plot financial metrics
    def plot_financial_metrics(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'], format='ISO8601')
        plt.figure(figsize=(14, 10))

        # Plot Close Prices and Moving Average
        plt.subplot(3, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
        plt.plot(self.data.index, self.data['Moving Average'], label='Moving Average', color='red')
        plt.title('Close Price and Moving Average')
        plt.legend()

        # Plot Daily Returns
        plt.subplot(3, 1, 2)
        plt.plot(self.data.index, self.data['Daily Returns'], label='Daily Returns', color='green')
        plt.title('Daily Returns')
        plt.legend()

        # Plot Rolling Volatility
        plt.subplot(3, 1, 3)
        plt.plot(self.data.index, self.data['Rolling Volatility'], label='Rolling Volatility', color='orange')
        plt.title('Rolling Volatility')
        plt.legend()

        plt.tight_layout()
        plt.show()