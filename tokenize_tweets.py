import csv
import nltk
nltk.download('punkt')

# Open the existing cleaned file and prepare to write to a new file
with open('consolidated_cleaned_data.csv', mode='r', newline='', encoding='utf-8') as infile, \
     open('tokenized_tweets.csv', mode='w', newline='', encoding='utf-8') as outfile:
    
    # Create CSV reader and writer objects
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)
    
    # Write header to the new file (include user_id)
    csv_writer.writerow(['author', 'tokenized_content'])  
    
    # Loop through each row in the CSV file
    for row in csv_reader:
        author = row[1]  # Assuming user_id is in the first column
        content = row[2]  # Assuming tweet text is in the 3rd column (index 2)
        
        # Tokenizing the tweet using NLTK
        nltk_tokens = nltk.word_tokenize(content)
        
        # Convert tokenized content back into a string
        tokenized_content = ' '.join(nltk_tokens)
        
        # Write user_id and tokenized tweet to the new file
        csv_writer.writerow([author, tokenized_content])

print("Tokenized tweets with user IDs have been written to 'tokenized_tweets.csv'")