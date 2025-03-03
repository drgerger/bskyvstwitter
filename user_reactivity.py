import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_user_reactivity(df):
    """
    Analyze user reactivity based on how extreme their valence, arousal, and dominance scores are.

    Parameters:
    df (pd.DataFrame): A DataFrame with at least:
                       - 'author' (user identifier)
                       - 'valence', 'arousal', 'dominance'

    Returns:
    pd.DataFrame: A summary ranking users by reactivity.
    """
    required_columns = {'author', 'valence', 'arousal', 'dominance'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    user_stats = df.groupby('author').agg(
        valence_mean=('valence', 'mean'),
        arousal_mean=('arousal', 'mean'),
        dominance_mean=('dominance', 'mean'),
        valence_std=('valence', 'std'),
        arousal_std=('arousal', 'std'),
        dominance_std=('dominance', 'std'),
        num_tweets=('valence', 'count')
    ).fillna(0)

    # Compute "reactivity score" as the average standard deviation of VAD scores
    user_stats['reactivity_score'] = (
        user_stats['valence_std'] + user_stats['arousal_std'] + user_stats['dominance_std']
    ) / 3

    return user_stats.sort_values(by='reactivity_score', ascending=False)

# Load the scored tweets dataset
tokenized_scored_tweets = pd.read_csv("tweets_with_vad_scores.csv")

# Compute user reactivity (author instead of user_id)
user_reactivity = compute_user_reactivity(tokenized_scored_tweets)

# Save to CSV
user_reactivity.to_csv("user_reactivity.csv", index=True)
print("User reactivity data has been saved to 'user_reactivity.csv'.")

# Display top 10 most reactive users
print("\nTop 10 Most Reactive Users:")
print(user_reactivity.head(10))

# Display top 10 least reactive users
print("\nTop 10 Least Reactive Users:")
print(user_reactivity.tail(10))

# Plot reactivity distribution
plt.figure(figsize=(10,6))
sns.histplot(user_reactivity['reactivity_score'], bins=30, kde=True)
plt.title("Distribution of User Reactivity Scores")
plt.xlabel("Reactivity Score")
plt.ylabel("Number of Users")
plt.show()

# Scatter plot of Reactivity Score vs Number of Tweets
plt.figure(figsize=(10,6))
sns.scatterplot(x=user_reactivity['num_tweets'], y=user_reactivity['reactivity_score'])
plt.title("User Reactivity vs Number of Tweets")
plt.xlabel("Number of Tweets")
plt.ylabel("Reactivity Score")
plt.show()