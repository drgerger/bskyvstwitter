import json
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer if needed
nltk.download('punkt')

# File paths for input JSONL files
input_files = [
    "mnt/data/posts_20241127_132451.jsonl",
    "mnt/data/posts_20241127_134301.jsonl"
]
consolidated_file = "mnt/data/consolidated_bluesky_posts.jsonl"
output_csv = "mnt/data/bluesky_vad_analysis.csv"

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
                vad_dict[word.lower()] = float(score)  # Convert score to float
    return vad_dict

# Load NRC-VAD lexicons
valence_dict = load_vad_lexicon(valence_file)
arousal_dict = load_vad_lexicon(arousal_file)
dominance_dict = load_vad_lexicon(dominance_file)

# Function to compute VAD scores for a given post
def get_vad_scores(text):
    if not isinstance(text, str):
        return 0, 0, 0  # Return zero scores if not a valid string

    words = word_tokenize(text.lower())  # Tokenize text and convert to lowercase
    valence = arousal = dominance = 0
    num_words = 0

    for word in words:
        if word in valence_dict or word in arousal_dict or word in dominance_dict:
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

# Function to consolidate JSONL files
def consolidate_jsonl(files, output_file):
    all_posts = []
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        post = json.loads(line.strip())  # Load JSON object
                        all_posts.append(post)
                        outfile.write(json.dumps(post) + "\n")
                    except json.JSONDecodeError:
                        print(f"Skipping malformed JSON line in {file_path}")
    return all_posts

# Function to process consolidated BlueSky dataset
def process_bluesky_dataset(input_file, output_csv):
    cleaned_data = []

    # Open and process the JSONL file (one JSON object per line)
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            try:
                post = json.loads(line.strip())  # Load JSON object
                author = post.get("author", "unknown")  # Get author ID
                text = post.get("text", "")  # Get post content
                
                # Tokenize the text
                tokenized_content = " ".join(word_tokenize(text))

                # Compute VAD scores
                valence, arousal, dominance = get_vad_scores(text)

                # Store processed data
                cleaned_data.append([author, tokenized_content, valence, arousal, dominance])
            except json.JSONDecodeError:
                print("Skipping malformed JSON line")

    # Convert to DataFrame
    df = pd.DataFrame(cleaned_data, columns=["author", "tokenized_content", "valence", "arousal", "dominance"])

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Processed data saved to {output_csv}")

# Consolidate BlueSky JSONL files into one
all_posts = consolidate_jsonl(input_files, consolidated_file)

# Process the consolidated dataset
process_bluesky_dataset(consolidated_file, output_csv)

# Display total number of posts processed
print(f"Total posts processed: {len(all_posts)}")
