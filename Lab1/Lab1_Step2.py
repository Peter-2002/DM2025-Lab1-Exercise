import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')


df = pd.read_csv("Reddit-stock-sentiment.csv")

print(df.head())
print(df.info())
print(df['label'].value_counts())

df = df[['text', 'label']]

df['category_name'] = df['label'].apply(lambda x: 'positive' if x==1 else ('neutral' if x==0 else 'negative'))
print(df['category_name'].value_counts())

for i, t in enumerate(df['text'][:5]):
    print(f"\nExample {i+1}:\n{t}")

plt.figure(figsize=(6,4))
sns.countplot(x='category_name', data=df)
plt.title("Category Distribution")
plt.show()


count_vect = CountVectorizer(stop_words='english')
X_counts = count_vect.fit_transform(df['text'])


feature_names = count_vect.get_feature_names_out()
doc_index = 4
word_indices = X_counts[doc_index].nonzero()[1]
words = [feature_names[i] for i in word_indices]
print(f"\nWords in document {doc_index}:\n{words}")

X_dense = X_counts.toarray()
df_tdm = pd.DataFrame(X_dense, columns=feature_names)

# Heatmap for a small sample
df_sample = df_tdm.sample(n=20, axis=0).iloc[:, :30]
plt.figure(figsize=(15, 8))
sns.heatmap(df_sample, cmap='YlGnBu', cbar=True)
plt.title("Heatmap of Sampled Term-Document Matrix")
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.show()

term_frequencies = np.asarray(X_counts.sum(axis=0)).ravel()
sorted_indices = np.argsort(term_frequencies)[::-1]
sorted_terms = feature_names[sorted_indices]
sorted_frequencies = term_frequencies[sorted_indices]

plt.figure(figsize=(20,6))
sns.barplot(x=sorted_terms[:50], y=sorted_frequencies[:50], color='blue')
plt.xticks(rotation=90)
plt.title("Top 50 Frequent Terms")
plt.show()

term_frequencies_log = np.log1p(term_frequencies)
sorted_frequencies_log = term_frequencies_log[sorted_indices]

plt.figure(figsize=(20,6))
sns.barplot(x=sorted_terms[:50], y=sorted_frequencies_log[:50], color='blue')
plt.xticks(rotation=90)
plt.title("Top 50 Frequent Terms (Log-scaled)")
plt.show()
