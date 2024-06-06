import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openpyxl

# Load English language model
nlp = spacy.load('en_core_web_sm')

# Read data from Excel
df = pd.read_excel('../data.xlsx')


# Tokenization and Lemmatization function
def tokenize_and_lemmatize(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct]
    return lemmatized_tokens


# Preprocess data
df['processed_text'] = df['name'] + ' ' + df['age'].astype(str) + ' ' + df['salary'].astype(str)

# Vectorize text data
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_lemmatize)
X = vectorizer.fit_transform(df['processed_text'])


# Training the model
def chatbot_response(input_text):
    input_text_processed = ' '.join(tokenize_and_lemmatize(input_text))
    input_vector = vectorizer.transform([input_text_processed])

    # Calculate cosine similarity between input and each data point
    similarity_scores = cosine_similarity(X, input_vector)

    # Get the index of the most similar data point
    most_similar_index = similarity_scores.argmax()

    # Get the corresponding response
    response = f"Name: {df.loc[most_similar_index, 'name']}, Age: {df.loc[most_similar_index, 'age']}, Salary: {df.loc[most_similar_index, 'salary']}"

    return response


# Example usage
input_question = "Give me name of person with 20000 salary"
response = chatbot_response(input_question)
print("Chatbot:", response)