import pandas as pd
import os
import yfinance as yf
import talib
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
        fig = px.line(
            self.data, 
            x='Date', 
            y='Close', 
            color='stock_symbol', 
            title='Close Price Over Time for Different Stock Symbols',
            labels={'Close': 'Price (USD)', 'Date': 'Date', 'stock_symbol': 'Stock Symbol'}
        )
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            legend_title='Stock Symbol',
            xaxis=dict(tickformat='%Y-%m-%d', tickangle=45)  # Format and rotate x-axis labels
        )
        
        fig.show()
        
        #Apply Analysis Indicators with TA-Lib 
    def calculate_moving_average(self, data, window_size):
        return talib.SMA(data, timeperiod=window_size)

    def calculate_technical_indicators(self, data):
        # Calculate various technical indicators
        data['SMA'] = self.calculate_moving_average(data['Close'], 20)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['EMA'] = talib.EMA(data['Close'], timeperiod=20)
        macd, macd_signal, _ = talib.MACD(data['Close'])
        data['MACD'] = macd
        data['MACD_Signal'] = macd_signal
        # Add more indicators as needed
        return data
    
        # Stock Price with Moving Average
    def plot_stock_data(self, data):
        fig = px.line(data, x=data.index, y=['Close', 'SMA'], title='Stock Price with Moving Average')
        fig.show()
        # Plot Relative Strength Index (RSI)
    def plot_rsi(self, data):
        fig = px.line(data, x=data.index, y='RSI', title='Relative Strength Index (RSI)')
        fig.show()

        # Stock Price with Exponential Moving Average
    def plot_ema(self, data):
        fig = px.line(data, x=data.index, y=['Close', 'EMA'], title='Stock Price with Exponential Moving Average')
        fig.show()
        
        # Plot Moving Average Convergence Divergence (MACD)
    def plot_macd(self, data):
        fig = px.line(data, x=data.index, y=['MACD', 'MACD_Signal'], title='Moving Average Convergence Divergence (MACD)')
        fig.show()
    
        # Calculate additional financial metrics using PyNance
    def calculate_financial_metrics(self, data):
        # Implement financial metric calculations here
        data['Returns'] = data['Close'].pct_change()
        # Additional metrics can be added here
        return data
    
        #Plot financial metrics
    def plot_financial_metrics(self, data):
        data = data.dropna()
        fig = px.line(data, x=data.index, y='Returns', title='Stock Returns Over Time')
        fig.show()
