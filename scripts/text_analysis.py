import re
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
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
    
    def format_date(self):
        self.data['date'] = pd.to_datetime(self.data['date'], format='ISO8601')
        return self.data['date']
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
    
    def analyze_publication_dates(self):
        """
        Analyze publication dates to identify trends over time.
        
        :return: Tuple of DataFrames - (articles_over_time, articles_by_day)
        """
        # call date format
        self.format_date()
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day_of_week'] = self.data['date'].dt.day_name()

        # Articles over time by year and month
        articles_over_time = self.data.groupby(['year', 'month']).size().reset_index(name='article_count')

        # Articles by day of the week
        articles_by_day = self.data['day_of_week'].value_counts()

        return articles_over_time, articles_by_day
    
    def plot_article_trends(self, articles_over_time, articles_by_day):
        """
        Plot trends of article counts over time and by day of the week.
        
        :param articles_over_time: DataFrame containing article counts over time.
        :param articles_by_day: Series containing article counts by day of the week.
        """
        # call date format
        self.format_date()
        # Plotting article counts over time
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=articles_over_time, x='month', y='article_count', hue='year', marker='o')
        plt.title('Article Count Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')
        plt.show()

        # Plotting article counts by day of the week
        plt.figure(figsize=(10, 5))
        sns.barplot(x=articles_by_day.index, y=articles_by_day.values)
        plt.title('Article Count by Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Articles')
        plt.show()

    # Time Series Analysis
    def analyze_time_series(self):
        """
        Perform time series analysis to understand publication frequency and spikes related to market events.
        
        :return: pd.Series with counts of publications by date and hour of publication.
        """
        # Convert 'date' to datetime if not already done
        if 'date' not in self.data or self.data['date'].dtype != 'datetime64[ns]':
            # call date format
            self.format_date()

        # Group by date to see publication frequency over time
        publication_freq = self.data['date'].value_counts().sort_index()

        # Extract hour from date to analyze publishing times
        self.data['hour'] = self.data['date'].dt.hour
        publishing_times = self.data['hour'].value_counts().sort_index()

        return publication_freq, publishing_times

    def plot_time_series_trends(self, publication_freq, publishing_times):
        """
        Plot the time series analysis results, including publication frequency and publishing times.
        
        :param publication_freq: pd.Series of publication counts by date.
        :param publishing_times: pd.Series of publication counts by hour.
        """
        # call date format
        self.format_date()
        # Plot publication frequency over time
        plt.figure(figsize=(12, 6))
        plt.plot(publication_freq.index, publication_freq.values, marker='o')
        plt.title('Publication Frequency Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        # Plot publishing times by hour
        plt.figure(figsize=(12, 6))
        sns.barplot(x=publishing_times.index, y=publishing_times.values, palette='viridis')
        plt.title('Publication Count by Hour of the Day')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.show()
        
        
        
        # Text Analysis
    def text_preprocess(self):
        # Convert to lowercase and remove non-alphabetic characters
        self.data['cleaned_headline']=self.data['headline'].str.lower().str.replace(r'[^a-zA-Z\s]', '', regex=True) 
        # Remove leading and trailing whitespace
        self.data['cleaned_headline']=self.data['cleaned_headline'].str.strip() 
        # remove stop words
        stop_words = set(stopwords.words('english'))
        self.data['cleaned_headline'] = self.data['cleaned_headline'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        return self.data
    
    def get_sentiment(self):
        # First preprocess the text
        self.text_preprocess()
        # Calculate the sentiment polarity of each headline
        self.data['polarity'] = self.data['cleaned_headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
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
        all_text = ' '.join(self.data['cleaned_headline']) 
        # Count the frequency of each word
        word_freq = pd.Series(all_text.split()).value_counts() 
        # Plot the top 20 most frequent words
        word_freq[:20].plot(kind='bar', figsize=(12, 6))
        plt.xticks(rotation=45)
        plt.xlabel('Words') 
        plt.ylabel('Frequency')
        plt.title('Word Frequency Distribution')
        plt.show()        
    
    # Keyword Extraction using TF-IDF
    def extract_keywords(self, n_keywords=5):
        # Initialize TF-IDF Vectorizer
        self.text_preprocess()
        vectorizer = TfidfVectorizer(max_features=n_keywords)
        tfidf_matrix = vectorizer.fit_transform(self.data['cleaned_headline'])
        
        # Extract keywords
        keywords = vectorizer.get_feature_names_out()
        return keywords
      
    # Topic Modeling using LDA
    def perform_topic_modeling(self, n_topics=2):
        self.text_preprocess()
        # Initialize TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(self.data['cleaned_headline'])
        
        # Perform LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(tfidf_matrix)
        
        # Display Topics
        words = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            topic_keywords = [words[i] for i in topic.argsort()[:-n_topics - 1:-1]]
            topics.append(f"Topic {topic_idx+1}: " + ", ".join(topic_keywords))
        
        return topics
    
    
    # Publisher Analysis
    def _extract_domain_from_email(self, email):
        """
        Extracts the domain from an email address.

        :param email: str, email address
        :return: str, domain extracted from the email
        """
        match = re.search(r'@([\w.-]+)', email)
        return match.group(1) if match else None

    def analyze_publishers(self):
        """
        Analyze the publisher data to determine the top publishers (emails) and domains,
        as well as publishers without domains.

        :return: tuple (pd.Series, pd.Series, pd.Series)
                 - publishers_with_domain: Frequency count of publishers with domains (emails).
                 - publishers_name: Frequency count of publishers without domains.
                 - publisher_domains: Frequency count of domains from publishers with emails.
        """
        # Extract domains from publishers
        self.data['domain'] = self.data['publisher'].apply(self._extract_domain_from_email)

        # Separate publishers with and without domains
        publishers_with_domain = self.data.dropna(subset=['domain'])
        publishers_name = self.data[self.data['domain'].isna()]

        # Count frequency of publishers with domains
        top_publishers_with_domain = publishers_with_domain['publisher'].value_counts()

        # Count frequency of publishers without domains
        top_publishers = publishers_name['publisher'].value_counts()

        # Count frequency of domains
        publisher_domains = publishers_with_domain['domain'].value_counts()

        return top_publishers_with_domain, top_publishers, publisher_domains

    def plot_publisher_analysis(self, publishers_with_domain, publishers_name, publisher_domains):
        """
        Plot analysis of publishers with and without domains, and their respective counts.

        :param publishers_with_domain: pd.Series of publishers name with domains and their article counts.
        :param publishers_name: pd.Series of publishers without domains and their article counts.
        :param publisher_domains: pd.Series of the domains extracted from publishers column.
        """
        # Plot publishers with domains
        plt.figure(figsize=(12, 6))
        sns.barplot(x=publishers_with_domain.index[:5], y=publishers_with_domain.values[:5], palette='coolwarm')
        plt.title('Top 5 Publishers with domain by Article Count')
        plt.xlabel('Publisher with Domain')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()

        # Plot publishers without domains
        plt.figure(figsize=(10, 6))
        sns.barplot(x=publishers_name.index[:5], y=publishers_name.values[:5], palette='coolwarm')
        plt.title('Top 5 Publishers by Article Count')
        plt.xlabel('Publisher Name')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()

        # Plot top domains
        plt.figure(figsize=(10, 6))
        sns.barplot(x=publisher_domains.index[:5], y=publisher_domains.values[:5], palette='coolwarm')
        plt.title('Top 5 Publisher Domains by Article Count')
        plt.xlabel('Domain name')
        plt.ylabel('Number of Articles')
        plt.xticks(rotation=45)
        plt.show()


    def save_data(self,columns):
        """
        Save the processed data to a specified file path.

        :param columns: str, list of column names to save the processed data.
        """
        self.data = self.data[columns]
        self.data.to_csv('final_news_data.csv', index=False)
        print("Processed data saved to 'final_news_data.csv'")