import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
import re
from nltk.stem import PorterStemmer


'''Downloading nltk punkt'''
# Attempt to disable SSL certificate verification (not recommended for production)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# Now try downloading the NLTK data
nltk.download('punkt')
nltk.download('stopwords')


'''Loading the data'''
df = pd.read_excel("/Users/amir/Desktop/own-interpreter/interpret/lib/python3.11/site-packages/mongodb data for clustering.xlsx")

print(df.head())


'''Tokenisation'''

if 'target' in df.columns:
    df['tokens'] = df['target'].apply(lambda x: word_tokenize(str(x)))
else:
    print("the text column does not exist in the dataset")
    
print(df['tokens'].head(20))


''' Stop Words'''

stop_words = set(stopwords.words('english'))

#Function to remove stop words
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]
    
#Applying the function to remove stopwords

if 'tokens' in df.columns:
    df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)
    
else:
    print("Column does not exist in the database")
    
print("The filtered tokens are, ", df['filtered_tokens'].head(20))


'''Removing Special Characters'''
def remove_special_characters(tokens):
    return [re.sub(r'[^\w\s]', '', word) for word in tokens]  # Removes all non-alphanumeric characters except spaces

if 'filtered_tokens' in df.columns:
    df['cleaned_tokens'] = df['filtered_tokens'].apply(remove_special_characters)
else:
    print("Column does not exist in the database")
print("Cleaned tokens: ", df['cleaned_tokens'].head(20))



''' Stemming '''
stemmer = PorterStemmer()
def apply_stemming(tokens):
    return [stemmer.stem(word) for word in tokens]

if 'cleaned_tokens' in df.columns:
    df['stemmed_tokens'] = df['cleaned_tokens'].apply(apply_stemming)
else:
    print("Column does not exist in the database")

# Display the processed data
print("Stemmed tokens: ", df['stemmed_tokens'].head(20))
