import pandas as pd

vad_df = pd.read_csv("mnt/data/bluesky_vad_analysis.csv")

# Compute summary statistics
print(vad_df[['valence', 'arousal', 'dominance']].describe())