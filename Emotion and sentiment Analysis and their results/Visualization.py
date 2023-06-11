import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
#
# df = pd.read_csv('clean.csv')
# vdf = df[['tweetContent_x']]
# text = ' '.join(vdf['tweetContent_x'].astype(str).tolist())
# wordcloud = WordCloud(width=600, height=600, background_color='white').generate(text)
# plt.figure(figsize=(8,8))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.tight_layout()
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.show()

# Load the data from CSV files into dataframes
df1 = pd.read_csv('ResultsRoberta.csv')
df2 = pd.read_csv('ResultsEmotion.csv')

# Select the columns for analysis
sentiments = df1['sentiment']
emotions = df2['sentiment']
print(sentiments.shape)
print(emotions.shape)

# Convert the columns to categorical data types
sentiments = pd.Categorical(sentiments, categories=[0, 1, 2])
emotions = pd.Categorical(emotions, categories=[0, 1, 2, 3, 4, 5, 6])

# Create a contingency table
ct = pd.crosstab(sentiments, emotions, rownames=['Sentiments'], colnames=['Emotions'])

# Create the heatmap using seaborn
sns.heatmap(ct, cmap='coolwarm', annot=True, fmt='d')

# Set the axis labels and title
plt.xlabel('Emotions')
plt.ylabel('Sentiments')
plt.title('Correlation between Sentiments and Emotions')
plt.show()
