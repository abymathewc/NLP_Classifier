import pandas as pd
import sys

from io import StringIO
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def get_text_count(data):
	"""
	Generate document term matrix by using scikit-learn's CountVectorizer.
	Using default n-gram size of (1,1)
	Stop words are removed, and sentences are converted to lower case .
	"""
	#tokenizer to remove unwanted elements from the  data like symbols and numbers
	token = RegexpTokenizer(r'[a-zA-Z0-9]+')
	cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
	text_counts= cv.fit_transform(data['sentence'])

	return text_counts

def train_model(X_train,y_train):
	"""
	create a Multinomial Naive Bayes classifier object using MultinomialNB() function.
	"""
	model = MultinomialNB().fit(X_train, y_train)
	return model


def main():
	#Read in the data using pandas and label the columns
	input_file = sys.argv[1]
	data = pd.read_csv(input_file,names=['sentence','label'])
	#print (data.head())
	#print (data.info())
	
	# Use Bag of words method to get text count. 
	text_counts = get_text_count(data)
	#print (text_counts)

	#split data into train & test
	X_train, X_test, y_train, y_test = train_test_split(text_counts, data['label'], test_size=0.8, random_state=1)

	#Train the model using fit method
	model = train_model(X_train,y_train)
	print ("Training complete....")

	#predict & verify the accuracy with the test data
	print("Validating with test data...")
	predicted =  model.predict(X_test)
	print("Multinomial Naive Bayes Accuracy:",metrics.accuracy_score(y_test, predicted))



if __name__ == "__main__":
    main()