import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import ssl

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



'''Loading the data'''
df = pd.read_excel("/Users/amir/Desktop/own-interpreter/interpret/lib/python3.11/site-packages/mongodb data for clustering.xlsx")

print(df.head())


'''Tokenisation'''

if 'target' in df.columns:
    df['tokens'] = df['target'].apply(lambda x: word_tokenize(str(x)))
else:
    print("the text column does not exist in the dataset")
    
print(df['tokens'].head(20))



