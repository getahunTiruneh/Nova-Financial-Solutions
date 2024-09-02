import pandas as pd
import os
import matplotlib.pyplot as plt
from textblob import TextBlob

class CorrelationAnalyzer:
    def __init__(self, file_path=None, folder_path=None):
        """
        Initializes the StockPriceAnalyzer class with a CSV file path or a folder path.
        
        param file_path: str (optional) - Path to a single CSV file containing stock data.
        param folder_path: str (optional) - Path to a folder containing multiple CSV files to be merged.
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
        
        folder_path: str - Path to the folder containing CSV files.
        return: DataFrame - Merged DataFrame of all CSV files in the folder.
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
    def date_alignment():
        pass
    def sentiment_analysis(self,text):
        analysis=TextBlob(text)
        return analysis.sentiment.polarity
    def calculate_daily_returns(self):
        pass
    def correlation_matrix(self):
        """
        Generates a correlation matrix heatmap for the stock data.

        return: matplotlib.figure.Figure - The correlation matrix heatmap.
        """
        # Select numeric columns for correlation calculation
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        
    def rename_column(self, old_name, new_name):
        if self.old_name in self.data.columns:
            data = data.rename(columns={old_name: new_name})
        return self.data