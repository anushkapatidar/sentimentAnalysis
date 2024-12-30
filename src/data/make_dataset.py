import pandas as pd 
import numpy as np
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import os 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def load_kaggle_sentiment_data(file_path):
    """
    Load and prepare sentiment data from Kaggle download
    Returns cleaned DataFrame ready for analysis
    """
    try:
        # Read the CSV file with specified column names
        try:
            df = pd.read_csv(file_path, 
                            encoding='utf-8',
                            names=['sentiment', 'id', 'timestamp', 'query', 'user', 'text'],
                            on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, 
                            encoding='latin-1',
                            names=['sentiment', 'id', 'timestamp', 'query', 'user', 'text'],
                            on_bad_lines='skip')
        
        # Print initial information
        print("\nDataset Overview:")
        print(f"Total number of records: {len(df)}")
        print("\nColumns in the dataset:")
        print(df.columns.tolist())
        
        # Display basic statistics
        print("\nBasic Statistics:")
        print(df.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        # Display first few rows
        print("\nFirst few rows of the dataset:")
        print(df.head())
        
        return df
    
    except Exception as e:
        print(f"Error loading the dataset: {str(e)}")
        return None

def prepare_sentiment_data(df):
    """
    Prepare the dataframe for sentiment analysis
    """
    # Create a clean copy
    prepared_df = df.copy()
    
    # Remove any rows with missing text
    prepared_df = prepared_df.dropna(subset=['text'])
    
    # Display the shape of prepared data
    print(f"\nPrepared dataset shape: {prepared_df.shape}")
    
    return prepared_df



# Function to clean the text
def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in STOPWORDS]
    return " ".join(text)


# Example usage
if __name__ == "__main__":
    # Example file path - replace with your actual path
    file_path = 'C:/Users/aakan/OneDrive/Desktop/anushka/sentimentAnalysis/data/training.csv'
    
    # Load the data
    df = load_kaggle_sentiment_data(file_path)
    
    if df is not None:
        # Prepare the data
        prepared_df = prepare_sentiment_data(df)

    if prepared_df is not None:
        # Save the prepared 
        # Drop unnecessary columns
        prepared_df = prepared_df.drop(columns=['id', 'timestamp', 'query', 'user'])

        # Apply cleaning to the 'text' column
        prepared_df['cleaned_text'] = prepared_df['text'].apply(clean_text)

        # # Drop rows with missing or empty text
        # prepared_df = prepared_df[prepared_df['text'].notnull() & (prepared_df['text'] != '')]

        # Remove duplicate rows if any
        prepared_df = prepared_df.drop_duplicates(subset=['cleaned_text'])

        # Save the cleaned dataset
        # prepared_df.to_csv('C:\Users\aakan\OneDrive\Desktop\anushka\sentimentAnalysis\data\cleaned_data\cleaned_dataset.csv')
        prepared_df.to_csv('C:\\Users\\aakan\\OneDrive\\Desktop\\anushka\\sentimentAnalysis\\data\\cleaned_data\\cleaned_dataset.csv', index=False)

        print("Dataset cleaned and saved to 'cleaned_dataset.csv'")
            



