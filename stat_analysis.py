from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# more statistical analysis of the scores

tokenized_scored_tweets = pd.read_csv("tweets_with_vad_scores.csv")

# Summary statistics of VAD scores
print(tokenized_scored_tweets[['valence', 'arousal', 'dominance']].describe())

tokenized_scored_tweets[['valence', 'arousal', 'dominance']].hist(bins=30, figsize=(12,6))
plt.suptitle("Distribution of Valence, Arousal, and Dominance Scores")
plt.show()

# classify valence
def classify_valence(val):
    if val > 0.7:
        return "Positive"
    elif val < 0.3:
        return "Negative"
    else:
        return "Neutral"

# Apply classification
tokenized_scored_tweets['sentiment_category'] = tokenized_scored_tweets['valence'].apply(classify_valence)

# Count of sentiment types
print(tokenized_scored_tweets['sentiment_category'].value_counts())

# Pie chart
tokenized_scored_tweets['sentiment_category'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6,6))
plt.title("Distribution of Sentiment Categories")
plt.show()

# Compute correlation matrix
correlation_matrix = tokenized_scored_tweets[['valence', 'arousal', 'dominance']].corr()

# Print correlation coefficients
print(correlation_matrix)

# # Heatmap
# plt.figure(figsize=(8,6))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation between Valence, Arousal, and Dominance")
# plt.show()

# # Group by topic and calculate mean sentiment
# topic_sentiment = tokenized_scored_tweets.groupby("topic")[["valence", "arousal", "dominance"]].mean()

# # Plot bar chart
# topic_sentiment.plot(kind="bar", figsize=(10,6))
# plt.title("Average Sentiment by Topic")
# plt.ylabel("Average Score")
# plt.xticks(rotation=45)
# plt.show()

# Use only VAD features
X = tokenized_scored_tweets[['valence', 'arousal', 'dominance']]

# Cluster tweets into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42)
tokenized_scored_tweets['cluster'] = kmeans.fit_predict(X)

# Scatter plot of clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=tokenized_scored_tweets['valence'], y=tokenized_scored_tweets['arousal'], hue=tokenized_scored_tweets['cluster'], palette="Set1")
plt.title("Clustering of Tweets Based on VAD Scores")
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.show()

# Assume engagement = likes + retweets + replies
tokenized_scored_tweets['engagement'] = tokenized_scored_tweets['likes'] + tokenized_scored_tweets['retweets'] + tokenized_scored_tweets['replies']

# Train model
X = tokenized_scored_tweets[['valence', 'arousal', 'dominance']]
y = tokenized_scored_tweets['engagement']
model = LinearRegression()
model.fit(X, y)

# Print feature importance
print("Model Coefficients:", dict(zip(X.columns, model.coef_)))