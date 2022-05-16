### Word to Vector



- You are given a set of product reviews on musical instruments that user brought from e-commerce website.
- You can find the dataset in reviews.json
- You are require to build word embeddings using gensim Word2Vec on the words present in user reviews.
- Follow the instruction provided in each cell to complete this task.



!pip install numpy
!pip install nltk
!pip install gensim



import json
import numpy as np
import nltk
import gensim




- create a list of reviews by reading reviews under 'reviewText' in reviews.json file
- assign the list to variable 'corpus'



f = open('reviews.json')
data = json.load(f)
corpus = []
for i in data:
corpus.append(i['reviewText'])
f.close()





### Text preprocessing
- For each review in corpus list perform following operation:
- convert the text to lowercase
- remove all non alpha characters
- remove all stop words from text, for list of stopwords refer nltk.corpus.stopwords
- remove all words not greater than or equal to two characters.
- tokenize the text
- perform lemmatization using nltk WordNetLemmatizer.



from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
new_corpus = []
for sent in corpus:
words = nltk.word_tokenize(sent)
words = [word.lower() for word in words]
words = [word for word in words if word.isalpha()]
words = [word for word in words if word not in stop_words]
shortword = re.compile(r'\W*\b\w{1,2}\b')
words = [word for word in words if(len(word)>2)]
words = [lemmatizer.lemmatize(word) for word in words ]
new_corpus.append(words)



### Generate Word embeddings
- Use preprocessed corpus to generate word embeddings using gensim.models.Word2Vec using following parameters:
- dimension: 300
- window size : 5
- min count : 5
- learning rate: 0.03
- number of negative samples: 5




from gensim.models import Word2Vec
model = Word2Vec(new_corpus, min_count=5,window=5, size=300, alpha=0.03,negative=5)
words = list(model.wv.vocab)




### Save the word embeddings in a csv file
- Each row in the file should have a word followed by its embedding values (refer sample.csv for correct format)
- save the csv file by name **embeddings.csv**



with open("embeddings.csv",'w') as fl:
for i in words:
#fl.write(i+" "+str(model[i]))
fl.write(i+","+','.join(map(str, model[i])))
fl.write("\n")