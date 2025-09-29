import numpy as np  
import pandas as pd 

df = pd.read_csv("spam.csv",encoding='latin-1')

# Data Cleaning

df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df = df.rename(columns={'v1': 'label', 'v2': 'text'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()
df = df.drop_duplicates(keep='first')

import nltk, string
nltk.download('punkt')

df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Data Preprocessing

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in nltk.corpus.stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(nltk.stem.PorterStemmer().stem(i))

  return " ".join(y)

df['transformed_text'] = df['text'].apply(transform_text)