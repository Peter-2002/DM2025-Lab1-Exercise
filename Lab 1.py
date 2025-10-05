# test code for environment setup
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt') # download the NLTK datasets
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import plotly as py
import math
# If you get "ModuleNotFoundError: No module named 'PAMI'"
# run the following in a new Jupyter cell:
# !pip3 install PAMI
import PAMI
import umap

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# obtain the documents containing the categories provided
from sklearn.datasets import fetch_20newsgroups

twenty_train = fetch_20newsgroups(subset='train', categories=categories,
                                  shuffle=True, random_state=42) 
#This command also shuffles the data randomly, but with random_state we can bring the same distribution of data everytime 
#if we choose the same number, in this case "42". This is good for us, it means we can reproduce the same results every time
#we want to run the code.
twenty_train.data[0:2]
twenty_train.target_names
len(twenty_train.data)
len(twenty_train.filenames)
# An example of what the subset contains
print("\n".join(twenty_train.data[0].split("\n")))
print(twenty_train.target_names[twenty_train.target[0]])
twenty_train.target[0]
# category of first 10 documents.
twenty_train.target[0:10]
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])
#Data Preparation
#Exercise 1
# Answer here
for i in range(3):
    print(f"example {i+1}")
    print("\n".join(twenty_train.data[i].split("\n")))

#Exercise 2
import pandas as pd

# my functions
import helpers.data_mining_helpers as dmh

# construct dataframe from a list
X = pd.DataFrame.from_records(dmh.format_rows(twenty_train), columns= ['text'])
len(X)
X[0:2]
for t in X["text"][:2]:
    print(t)
#Adding Columns
# add category to the dataframe
X['category'] = twenty_train.target
# add category label also
X['category_name'] = X.category.apply(lambda t: dmh.format_labels(t, twenty_train))
X[0:10]
#Exercise 2
#Answer here
X[X['category_name'] == 'sci.med'][0:10]
#Exercise 3
# Answer here
X[X["category_name"]=="sci.med"][::10][0:5]
#Data mining Using Pandas
#Exercise 4
### >>> **Exercise 4 (Watch Video):** 
'''Let's try something different. Instead of calculating missing values by column let's try to calculate the missing values in every record instead of every column.  
$Hint$ : `axis` parameter. Check the documentation for more information.'''
# Answer here
X.isnull().apply(lambda x: dmh.check_missing_values(x), axis=1)
#Can skip Exercise 5
#Duplicate Data
#Exercise 6
# Answer
#X_sample reduced to 1000 from all rows which only some rows are kept. The row in X_sample is not sequenqial
#The index values are preserved from the dataframe, so they appear non-sequential
#The order of rows is shuffled due to randomsampling
#The numbers of rows decreased from full set to 1000,but the columns like(text, category and category name) remain unchanged
#so overall data reamin unchanged just reduced in number of datasets
#Exercise 7
# Answer here
X_sample.category_name.value_counts().plot(kind = 'bar', title = 'Category distribution', 
                                           ylim = [0, X_sample.category_name.value_counts().max()+30],
                                           rot = 0, fontsize = 12, figsize = (8,3))
#Exercise 8
# Answer here
# counts for each category in the full dataset and the sample
counts_full   = X['category_name'].value_counts()
counts_sample = X_sample['category_name'].value_counts().reindex(counts_full.index, fill_value=0)

# combine into one DataFrame so pandas plots grouped bars
df_counts = pd.concat([counts_full, counts_sample], axis=1)
df_counts.columns = ['full', 'sample']

# plot
ax = df_counts.plot(kind='bar', figsize=(10,5), rot=0, fontsize=12, title='Category distribution')
ax.set_ylabel('count')
ax.set_ylim(0, df_counts.values.max() + 30)
ax.legend(['category_name', 'category_name'])
plt.tight_layout()
plt.show()
#exercise 9
# Answer here
# How do we turn our array[0] text document into a tokenized text using the build_analyzer()?
analyze(X.text[0])
#Exercise 10
# Answer here
feature_names = count_vect.get_feature_names_out()
doc_index = 4
word_indices = X_counts[doc_index].nonzero()[1]
words = [feature_names[i] for i in word_indices]
print(words)

### **>>> Exercise 11 (take home):** 
#From the chart above, we can see how sparse the term-document matrix is; i.e.,
#there is only one terms with **FREQUENCY** of `1` in the subselection of the matrix. 
#By the way, you may have noticed that we only selected 20 articles and 20 terms to plot the histrogram. 
#As an excersise you can try to modify the code above to plot the entire term-document matrix 
#or just a sample of it. How would you do this efficiently? Remember there is a lot of words in the vocab. 
#Report below what methods you would use to get a nice and useful visualization
# Answer here
# Randomly sample 100 documents and 200 terms
X_plot = (X_counts > 0).sum(axis=1).A1
plt.hist(X_plot, bins=50)
plt.title("Non-zero Distribution per document")
plt.xlabel("Terms")
plt.ylabel("Sets of Document")
plt.show()
X_plot
#Exercise 12
# Answer here
X_dense = X_counts.toarray()

# Create a DataFrame with feature names as column headers
df_tdm = pd.DataFrame(X_dense, columns=count_vect.get_feature_names_out())

# To make it readable, sample a subset (e.g., 20 documents × 30 terms)
df_sample = df_tdm.sample(n=20, axis=0)  # 20 random documents
df_sample = df_sample.iloc[:, :30]        # 30 most frequent terms

# Plot the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(df_sample, cmap='YlGnBu', cbar=True)
plt.title("Heatmap of Sampled Term-Document Matrix")
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.show()
#Exercise 13
### >>> **Exercise 13 (take home):** 
#The chart above only contains 300 vocabulary in the documents, 
# and it's already computationally intensive to both compute and visualize. 
# Can you efficiently reduce the number of terms you want to visualize as an exercise. 

# Answer here
plt.subplots(figsize=(100, 10))
g = sns.barplot(x=count_vect.get_feature_names_out()[:50], 
            y=term_frequencies[:50])
g.set_xticklabels(count_vect.get_feature_names_out()[:50], rotation = 90)
#Exercise 14
#Sort by frequency in X-Axis
# Answer here
#sort the term by frequency not by alphabetical order wพรte the coกe to sort it
term_frequencies = np.asarray(X_counts.sum(axis=0)).ravel()
terms = count_vect.get_feature_names_out()
sorted_indices = np.argsort(term_frequencies)[::-1]
sorted_terms = terms[sorted_indices]
sorted_frequent = term_frequencies[sorted_indices]
plt.figure(figsize=(20,6))
g = sns.barplot(x=sorted_terms[:300], y=sorted_frequent[:300],color='blue')
g.set_xticklabels(sorted_terms[:300], rotation=90)
g.set_title("Top Frequent Terms by frequency")
plt.xlabel("Terms")
plt.ylabel("Frequency")
plt.show()

#Exercise 15
# Answer here
term_frequencies = np.asarray(X_counts.sum(axis=0)).ravel()
term_frequencies_log = np.log1p(term_frequencies)

# sort by log frequency
sorted_indices = np.argsort(term_frequencies_log)[::-1]
sorted_terms = count_vect.get_feature_names_out()[sorted_indices]
sorted_frequencies_log = term_frequencies_log[sorted_indices]

plt.figure(figsize=(20, 6))
sns.barplot(x=sorted_terms[:300], y=sorted_frequencies_log[:300], color='blue')
plt.xticks(rotation=90)
plt.title("Top 300 Most Frequent Terms (Log-Scaled)")
plt.xlabel("Terms")
plt.ylabel("Log(1 + Frequency)")
plt.show()

#The Different showsthe reduction of extremely frequent words and make the word distribution easier to see
#and interpret. The log keep the ranking the same but compresses large/extreme values
#helping highlight less frequent but meaningful. terms
