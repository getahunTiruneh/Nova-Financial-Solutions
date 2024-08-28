import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TextAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data

    def check_data_quality(self):
        # Check for missing values
        missing_values = self.data.isnull().sum()
        print("Missing values:")
        print(missing_values)

        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        print("\n Number of duplicates:", duplicates)

        # Check data types
        data_types = self.data.dtypes
        print("\n Data types:")
        print(data_types)
        
        # Drops the 'Unnamed' column from a pandas DataFrame
    def drop_unnamed_column(self):
        
        self.data = self.data.drop(['Unnamed: 0'], axis=1)
        return self.data
    
        # Descriptive Statistics
    def headline_length_stats(self):
        
        self.data['headline_length'] = self.data['headline'].str.len()
        print("Headline Length Statistics:")
        return self.data['headline_length'].describe()
    
        # Number of Articles per publisher 
    def article_per_publisher(self):
        publisher_counts = self.data['publisher'].value_counts().to_frame().reset_index()
        publisher_counts.columns = ['Publisher', 'Article Count']
        print("Article Counts per Publisher:")
        return publisher_counts
    
    def plot_article_per_publisher(self):
        publisher_counts = self.article_per_publisher().nlargest(5, 'Article Count')
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Publisher', y='Article Count', data=publisher_counts)
        plt.xticks(rotation=90)
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.title('Number of Articles per Publisher')
        plt.show()
    
    def publication_dates_analysis(self):
        self.data['date']=pd.to_datetime(self.data['date'],format='ISO8601')
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        
        articles_per_day=self.data.groupby(['year', 'month', 'day']).size().reset_index(name='article_count')
        articles_per_day.plot(x='day', y='article_count', kind='line', figsize=(15, 5))
        plt.xlabel('Day of the Month')
        plt.ylabel('Number of Articles')
        plt.title('Number of Articles per Day of the Month')
        plt.show()
    # Time Series Analysis
    def publication_frequency_over_time(self): # Group the data by year and month, and count the number of articles for each group
        self.publication_dates_analysis()
        articles_per_month = self.data.groupby(['year', 'month']).size()
        articles_per_month.plot() # Plot the time series data
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.title('Articles per Month')
        plt.show()
       
    def unique_domains_from_emails(self):
        if self.data['publisher'].str.contains('@').any(): 
            self.data['domain'] = self.data['publisher'].apply(lambda x: x.split('@')[-1] if '@' in x else None) # Extract the domain from the email addresses
            return self.data['domain'].value_counts()
        
        
        
        # Text Analysis
    def text_preprocess(self):
        self.drop_unnamed_column()
        # Convert to lowercase and remove non-alphabetic characters
        self.data['cleand_headline']=self.data['headline'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True) 
        # Remove leading and trailing whitespace
        self.data['cleand_headline']=self.data['cleand_headline'].str.strip() 
        # remove stop words
        stop_words = set(stopwords.words('english'))
        self.data['cleand_headline'] = self.data['cleand_headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        return self.data
    
    def get_sentiment(self):
        # First preprocess the text
        self.text_preprocess()
        # Calculate the sentiment polarity of each headline
        self.data['polarity'] = self.data['cleand_headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
        # Categorize the sentiment based on the polarity score
        self.data['sentiment'] = self.data['polarity'].apply(lambda x: 'positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
        return self.data
    def plot_sentiment_distribution(self):
        # First preprocess the text
        self.text_preprocess()
        
        # Visualize the sentiment distribution
        sentiment_counts = self.get_sentiment()['sentiment'].value_counts()
        print(sentiment_counts)  # Print the counts for each sentiment category
        plt.figure(figsize=(8, 6)) # Set the figure size
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)  # Create a bar plot
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.xlabel('Sentiment')  # Set the x-axis label
        plt.ylabel('Number of Articles')  # Set the y-axis label
        plt.title('Sentiment Distribution')  # Set the title of the plot
        plt.show()  # Display the plot
    
    def word_frequency(self):
        # First preprocess the text
        self.text_preprocess()
        # Concatenate all headlines into a single string
        all_text = ' '.join(self.data['cleand_headline']) 
        # Count the frequency of each word
        word_freq = pd.Series(all_text.split()).value_counts() 
        # Plot the top 20 most frequent words
        word_freq[:20].plot(kind='bar', figsize=(12, 6))
        plt.xticks(rotation=45)
        plt.xlabel('Words') 
        plt.ylabel('Frequency')
        plt.title('Word Frequency Distribution')
        plt.show()        
        
    def topic_modeling(self, n_topics=5):
        # First preprocess the text
        self.text_preprocess()
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform(self.data['cleand_headline'])
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(X)
    
        def print_top_words(model, feature_names, n_words=10):
            for topic_idx, topic in enumerate(model.components_):
                print(f"Topic {topic_idx}:")
                print(" ".join([feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]))

        return print_top_words(lda, vectorizer.get_feature_names_out())