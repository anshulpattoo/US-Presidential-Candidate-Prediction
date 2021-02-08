import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

brown = nltk.corpus.brown

#Imports
import unicodedata
import re
import string
from nltk.tokenize import ToktokTokenizer
import contractions
import pandas as pd
import numpy as np
import sklearn



file  = open('negative-words.csv', "r")
speechRows = file.readlines()
speeches = []



#Remove punctuation.

# Function to remove accented characters
def remove_accented_chars(text):
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text


#Remove negwords.

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    # convert sentence into token of words
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    # check in lowercase 
    t = [token for token in tokens if token.lower() not in stopword_list]
    text = ' '.join(t)    
    return text

#Remove special characters.

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

#Remove numbers.

def remove_numbers(text):
    # define the pattern to keep
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    return re.sub(pattern, '', text)

#Remove punctuation.

def remove_punctuation(text):
    under = ['_']
    punc = set(string.punctuation) - set(under)
    text = ''.join([c for c in text if c not in punc])
    return text


#Generate the list of speeches.
fileTwo  = open('speeches.csv', "r")
fileTwoLines = fileTwo.readlines()
speechNames = []
for line in fileTwoLines:
  speechNames.append(line.rstrip('\n').rstrip('.txt'))

#Generate the list of winners.
fileThree = open('winners.csv', "r")
fileThreeLines = fileThree.readlines()
winners = []
for line in fileThreeLines:
  winners.append(int(line))

nonWords = ['al', 'st', 'mr']

#Generate the list of words
with open('mostfreq1000word.csv', 'r', errors='replace') as f:
  lines = f.readlines()
indices = []
wordsPOS = []
i = 0
for line in lines:
  word = line[0:line.index('_')]
  #Separate tag and THEN add the tag after.
  word = remove_stopwords(word)
  word = contractions.fix(word)
  word = remove_punctuation(word)
  word = remove_special_characters(word)
  word = remove_accented_chars(word)
  
  if (word == '' or word in nonWords):
    indices.append(i)
  else:
    line = word + line[line.index('_'):line.index('\n')]
    wordsPOS.append(line)

  i += 1

file  = open('mostfreq1000docword.csv', "r")
speechRows = file.readlines()
speeches = []

counter = 0
for row in speechRows:
  speech = row.rstrip('\n').split(',')
  speechNums = [float(i) for i in speech]
  for index in sorted(indices, reverse=True):
    del speechNums[index]
  speechNums.append(winners[counter])
  speeches.append(speechNums)
  counter += 1
columns = wordsPOS
columns.append('winners')


speeches = np.array(speeches)

#speeches_std = (speeches - speeches.min(axis=0)) / (speeches.max(axis=0) - speeches.min(axis=0))
#speeches_scaled = speeches_std * (1 - 0) + 0

df = pd.DataFrame(np.array(speeches), columns = columns)
df.to_csv('mergedmodified.csv')

#Comparing original data with current data.



counter = 0
speechesOriginal = []
for row in speechRows:
  speech = row.rstrip('\n').split(',')
  speechNums = [float(i) for i in speech]
  speechNums.append(winners[counter])
  speechesOriginal.append(speechNums)
  counter += 1

columnsOriginal = []
for line in lines:
  line = line.rstrip('\n')
  columnsOriginal.append(line)
columnsOriginal.append('winners')


df = pd.DataFrame(np.array(speechesOriginal), columns = columnsOriginal)
print(df)
df.to_csv('mergedoriginal.csv')


