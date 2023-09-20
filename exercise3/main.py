import nltk
import matplotlib.pyplot as plt
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('stopwords')
nltk.download('omw-1.4')
gutenberg_corpus = gutenberg.open('melville-moby_dick.txt')

# #Reading Moby Dick files
text = gutenberg.raw('melville-moby_dick.txt')

# participle
tokens = word_tokenize(text)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Part-of-speech tagging
tagged_tokens = pos_tag(filtered_tokens)

# Part of speech frequency statistics
freq_dist = FreqDist(tag for word, tag in tagged_tokens)
top5_tags = freq_dist.most_common(5)
print("Top 5 POS Tags:")
for tag, count in top5_tags:
    print(f"{tag}: {count}")

# Word Form Merge
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word, tag in tagged_tokens[:20]]
print("Lemmatized Words:")
print(lemmatized_words)

# Draw a part of speech distribution chart
all_tags = [tag for word, tag in tagged_tokens]
tag_freq = FreqDist(all_tags)
tag_freq.plot()

# Sentiment analysis
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
avg_sentiment_score = sentiment_scores['compound']
print("Average Sentiment Score:", avg_sentiment_score)
if avg_sentiment_score > 0.05:
    print("Overall Sentiment: Positive")
else:
    print("Overall Sentiment: Negative")
