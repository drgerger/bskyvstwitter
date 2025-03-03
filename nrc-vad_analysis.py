import pandas as pd
import matplotlib.pyplot as plt

# File paths for the NRC-VAD lexicon files
valence_file = "mnt/data/valence-NRC-VAD-Lexicon.txt"
arousal_file = "mnt/data/arousal-NRC-VAD-Lexicon.txt"
dominance_file = "mnt/data/dominance-NRC-VAD-Lexicon.txt"

# Function to load NRC-VAD lexicon into a dictionary
def load_vad_lexicon(file_path):
    vad_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")  # Split word and score
            if len(parts) == 2:
                word, score = parts
                vad_dict[word.lower()] = float(score)
    return vad_dict

# Load VAD lexicons into dictionaries
valence_dict = load_vad_lexicon(valence_file)
arousal_dict = load_vad_lexicon(arousal_file)
dominance_dict = load_vad_lexicon(dominance_file)

# Function to compute VAD scores for a given tweet
def get_vad_scores(tweet):
    if not isinstance(tweet, str):
        return 0, 0, 0  # Return zero scores if not a string

    words = tweet.split()
    valence = arousal = dominance = 0
    num_words = 0

    for word in words:
        word = word.lower()
        valence += valence_dict.get(word, 0)
        arousal += arousal_dict.get(word, 0)
        dominance += dominance_dict.get(word, 0)
        num_words += 1

    # Compute average scores
    if num_words > 0:
        valence /= num_words
        arousal /= num_words
        dominance /= num_words

    return valence, arousal, dominance

# Load the tokenized tweets dataset (ensure it has 'user_id')
tokenized_tweets = pd.read_csv("tokenized_tweets.csv")

# Apply VAD score computation
tokenized_tweets[['valence', 'arousal', 'dominance']] = tokenized_tweets['tokenized_content'].apply(get_vad_scores).apply(pd.Series)

# Save results including user_id
tokenized_tweets.to_csv("tweets_with_vad_scores.csv", index=False)

print("VAD scores have been computed and saved with user IDs.")
