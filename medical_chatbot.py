# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 03:19:08 2025

@author: Harris Popal
"""

import streamlit as st
import pandas as pd
import re
import nltk
import spacy
import numpy as np
import os
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model for Named Entity Recognition (NER)
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Small sample dataset
df = pd.read_csv('train_data.csv')

df['Description'] = df['Description'].str.replace(r'^Description:\s*', '', regex=True)
df['Description'] = df['Description'].str.replace(r'^Q\.\s*', '', regex=True)
df = df[['Patient', 'Description', 'Doctor']]
df = df.drop_duplicates(subset=['Patient'])

number_pattern = re.compile(r'\d+')
special_char_pattern = re.compile(r'\W+')

STOP_WORDS = set(stopwords.words('english')).union({
    'hi', 'hello', 'doctor', 'year', 'old', 'thanks', 'yrs', 'suggest', 'remedy', 'treatment',
    'years', 'patient', 'patients', 'dear', 'sir', 'thank', 'cure', 'reason', 'treated', 'cause',
    'age', 'name', 'doc', 'please', 'help', 'causes', 'suggestion', 'could', 'want', 'treat'
})

def clean_text(text):
    text = text.lower()
    text = number_pattern.sub('', text)  # Remove numbers
    text = special_char_pattern.sub(' ', text)  # Remove special characters
    words = (w for w in text.split() if w not in STOP_WORDS)
    return " ".join(words)

def deduplicate_tokens(text):
    tokens = text.split()
    return " ".join(dict.fromkeys(tokens))

def extract_medical_entities(text):
    doc = nlp(text)
    medical_labels = {"DISEASE", "SYMPTOM", "CONDITION", "MEDICATION"}  # Define entity labels of interest
    entities = [ent.text for ent in doc.ents if ent.label_ in medical_labels]
    extracted = " ".join(entities) if entities else clean_text(text)  # Fallback to cleaned text if no entities
    return deduplicate_tokens(extracted)

df['Clean_Symptoms'] = df['Patient'].apply(extract_medical_entities)
df['Clean_Description'] = df['Description'].apply(extract_medical_entities)

df = pd.read_csv('processed_data.csv')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

@st.cache_resource
def load_sbert_model():
    # Load the medical-specific SentenceTransformer model.
    return SentenceTransformer('pritamdeka/S-PubMedBERT-MS-MARCO')

sbert_model = load_sbert_model()

@st.cache_data
def get_train_embeddings():
    embedding_file = "train_desc_embeddings.npy"
    if os.path.exists(embedding_file):
        # st.write("Loading precomputed embeddings from file...")
        embeddings = np.load(embedding_file)
    else:
        st.write("Computing embeddings, this may take a while...")
        train_desc_texts = train_df['Clean_Description'].tolist()
        embeddings = sbert_model.encode(train_desc_texts, show_progress_bar=False)
        np.save(embedding_file, embeddings)
        st.write(f"Embeddings saved to {embedding_file}")
    return embeddings

train_desc_embeddings = get_train_embeddings()

@st.cache_data
def retrieve_best_candidate(symptoms_text, top_k=10):
    # Encode the input symptoms text.
    input_emb = sbert_model.encode([symptoms_text], show_progress_bar=False)
    # Compute cosine similarity between the query and all training embeddings.
    sims = cosine_similarity(input_emb, train_desc_embeddings)[0]
    # Retrieve indices for the top_k most similar descriptions.
    top_k_idx = np.argsort(sims)[-top_k:][::-1]
    best_idx = top_k_idx[0]
    return (train_df.iloc[best_idx]['Clean_Description'],
            train_df.iloc[best_idx]['Doctor'],
            sims[best_idx])

@st.cache_data
def evaluate_test_set():
    test_symptoms = test_df['Clean_Symptoms'].tolist()
    # Batch encode all test symptoms.
    test_emb = sbert_model.encode(test_symptoms, show_progress_bar=False)
    # Compute cosine similarity matrix between test and training embeddings.
    sims_matrix = cosine_similarity(test_emb, train_desc_embeddings)
    # For each test sample, get the maximum similarity score.
    best_similarities = np.max(sims_matrix, axis=1)
    return best_similarities

test_similarities = evaluate_test_set()
test_df['Similarity_Score'] = test_similarities
st.write(f"Optimized Retrieval-Based Average Cosine Similarity: {test_df['Similarity_Score'].mean():.4f}")


def chatbot():
    st.title("🩺 Medical Chatbot")
    st.write("Enter your symptoms, and I'll retrieve a similar case description along with a doctor's advice.")

    user_input = st.text_area("Describe your symptoms:")

    if st.button("Get Diagnosis"):
        if user_input:
            # Extract medical entities from the user input.
            filtered_input = extract_medical_entities(user_input)
            # Retrieve the best candidate based on the filtered input.
            retrieved_desc, doctor_response, similarity_score = retrieve_best_candidate(filtered_input, top_k=10)

            st.subheader("Closest Matched Medical Description:")
            st.write(retrieved_desc)

            st.subheader("Doctor's Response:")
            st.write(doctor_response)

            st.subheader("Similarity Score:")
            st.write(f"{similarity_score:.4f}")
        else:
            st.write("Please enter your symptoms.")

if __name__ == "__main__":
    chatbot()