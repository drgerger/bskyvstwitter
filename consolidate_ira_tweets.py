import pandas as pd
import glob
import os
import re
import string

directory_path = os.path.abspath('/Users/dessagerger/social_media_sentiment_project/twitter/russian-troll-tweets-master')
files = os.listdir(directory_path)
print(f"Files in directory: {files}")
csv_files = [file for file in files if file.endswith('.csv')]

cleaned_dataframes = []

print(f"Current working directory: {os.getcwd()}")

# function to remove URLs from the 'content' column
def clean(text):
    # Check if the text is a string before applying the regex
    if isinstance(text, str):
        # Regular expression pattern to match URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        # Regular expression pattern to match hashtags
        hashtag_pattern = r'#\w+'
        # Regular expression pattern to match numbers
        number_pattern = r'\d+'
        # Regular expression pattern to match punctuation
        punctuation_pattern = r'[' + re.escape(string.punctuation) + r']'
        
        # Remove URLs
        text = re.sub(url_pattern, '', text)
        # Remove hashtags
        text = re.sub(hashtag_pattern, '', text)
        # Remove numbers
        text = re.sub(number_pattern, '', text)
        # Remove punctuation
        text = re.sub(punctuation_pattern, '', text)
        
        return text
    return text  # If it's not a string, return it as is

for file in csv_files: # processing each individual csv
    print("file name: ", file)
    file = 'russian-troll-tweets-master/' + file
    df = pd.read_csv(file)
    
    # filter rows where the language is 'English'
    df_cleaned = df[df['language'] == 'English']
    df_cleaned['content'] = df_cleaned['content'].apply(clean)
    
    # append the cleaned DataFrame to the list
    cleaned_dataframes.append(df_cleaned)

# concatenate all cleaned DataFrames into one
consolidated_cleaned_df = pd.concat(cleaned_dataframes, ignore_index=True)

# save the consolidated cleaned data to a new CSV file
consolidated_cleaned_df.to_csv('consolidated_cleaned_data.csv', index=False)

print("All CSV files have been cleaned and consolidated into 'consolidated_cleaned_data.csv'")
