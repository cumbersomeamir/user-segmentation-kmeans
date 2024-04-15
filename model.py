import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl
import re
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.cluster import KMeans
from openai import OpenAI
import os


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
df = pd.read_excel("/Users/amir/Desktop/own-interpreter/interpret/lib/python3.11/site-packages/mongodb data for clustering sample.xlsx")

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

# Combine tokens into a single string for embeddings
df['text_for_embedding'] = df['stemmed_tokens'].apply(lambda tokens: ' '.join(tokens))

# Attempt to read the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
client = OpenAI()

# Fetch embeddings
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

df['embedding'] = df['text_for_embedding'].apply(get_embedding)

# Prepare embeddings for clustering
embeddings_matrix = np.array(list(df['embedding']))

# K-means clustering
n_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(embeddings_matrix)

# Print results
print(df[['text_for_embedding', 'cluster']].head())

# Export DataFrame to Excel
output_file_path = "clustered_dataset.xlsx"  # Specify your output file path
df.to_excel(output_file_path, index=False)

print(f"Data exported to {output_file_path} successfully!")

'''Visualisation'''
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# First, we reduce the dimensionality of the embeddings to 2D for visualization
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings_matrix)

# Scatter plot of the reduced data with cluster assignments
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.title('Cluster visualization using PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(scatter)
plt.show()
