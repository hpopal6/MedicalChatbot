# AI-Powered Medical Chatbot

<img width="749" alt="Image" src="https://github.com/user-attachments/assets/9bf5d200-a6aa-4d43-ae2d-9cb46de7d53f" />

## Overview

This project showcases an AI-powered medical chatbot designed to assist users in understanding their symptoms. By leveraging machine learning and natural language processing (NLP), the chatbot aims to bridge the gap between patients and medical information, potentially reducing reliance on unreliable online sources and encouraging timely medical consultations.

Our exploration delved into various ways of working with medical data, culminating in this chatbot and interactive visualizations. We also integrated multiple independent datasets to explore medical disease and survival prediction.

## Motivation

Patients often face challenges in interpreting their symptoms. This can lead to:

* **Delayed Medical Consultations:** Uncertainty about symptoms might deter individuals from seeking timely professional help.
* **Reliance on Unreliable Online Sources:** The internet, while a vast resource, can often provide inaccurate or anxiety-inducing medical information.

This project aims to address these issues by providing an accessible and informative AI-powered tool. Furthermore, we were driven by the desire to explore the potential of machine learning and NLP in the medical domain, specifically in creating interactive tools and integrating diverse medical datasets.

## Hypothesis

We hypothesized that a Natural Language Processing (NLP) model trained on real doctor-patient interactions could provide meaningful diagnostic insights based on user-provided symptom descriptions.

## Datasets Used

### Patient-Doctor Chatbot Dataset (Hugging Face)

* **Source:** [Patient-Doctor Chatbot Dataset](https://huggingface.co/datasets/health_care_chatbot)
* **Description:** This dataset contains 154,150 real-world patient-doctor dialogues, encompassing patient symptoms, doctor diagnoses, and treatment suggestions.
* **Usage:** This dataset was crucial for training our NLP model to understand and predict potential medical conditions based on user-inputted symptoms.

## Data Preprocessing

### Textual Data (Chatbot)

* **Preprocessing:** We performed several preprocessing steps on the textual data from the chatbot dataset, including:
    * **Removal of Stopwords:** Eliminating common words (e.g., "the," "a," "is") that do not contribute significantly to the meaning.
    * **Named Entity Recognition (NER):** Identifying and extracting medical entities, particularly symptoms, using libraries like SpaCy.
* **Feature Engineering:** To prepare the text data for machine learning models, we employed:
    * **TF-IDF Vectorization:** Converting text data into numerical vectors based on the term frequency-inverse document frequency, allowing for text classification.

## Technologies and Tools

* **Programming Languages:**
    * Python (Pandas, Scikit-learn, TensorFlow, PyTorch)
* **NLP Model:**
    * Sentence-BERT (SBERT)
* **Machine Learning Algorithms:**
    * Logistic Regression
    * Random Forest
    * CatBoost
* **APIs & Libraries:**
    * Hugging Face Transformers
    * SpaCy (for Named Entity Recognition)
    * Streamlit (for creating the web application interface)
* **Development Environment:**
    * Google Colab
    * Jupyter Notebook

## Model Implementation

### Medical Chatbot (NLP)

Our medical chatbot leverages NLP techniques to understand user input and provide relevant information:

1.  **Initial Classification (TF-IDF + Logistic Regression):** We initially employed a combination of TF-IDF vectorization and a Logistic Regression model for a baseline classification of user queries.
2.  **Similarity Retrieval (Sentence-BERT):** For more nuanced understanding and retrieval of relevant doctor-patient dialogues, we utilized Sentence-BERT (SBERT) embeddings. SBERT allows us to calculate the semantic similarity between the user's input and the dialogues in our training data, enabling the chatbot to provide contextually relevant responses.
3.  **Streamlit UI:** We developed a user-friendly web interface using Streamlit, allowing users to easily interact with the chatbot by inputting their symptoms and receiving potential insights.

## Medical Chatbot Performance

* **Sentence-BERT Similarity Matching Score:** The Sentence-BERT model achieved an average cosine similarity score of **0.91** when matching user input to relevant dialogues in the dataset. This indicates a high degree of semantic similarity in the retrieved responses.

### User Input Example:

**User Input:** "I have persistent headaches and nausea."

**Predicted Diagnosis:** Migraine or High Blood Pressure

**Doctor’s Advice (from similar dialogue):** Monitor hydration and check blood pressure.

## Implications of Findings

The successful implementation of our NLP chatbot demonstrates the potential of using machine learning to simulate aspects of a doctor-patient consultation. The high similarity scores achieved by Sentence-BERT suggest that the model can effectively understand the semantic meaning of user-provided symptoms and retrieve relevant information from real-world medical dialogues.

However, it's important to note that the current model may require additional training on more complex and rare medical conditions to broaden its knowledge base and improve its accuracy across a wider range of scenarios.

## Comparison with Existing Research

Our approach, particularly the use of Sentence-BERT embeddings, represents an improvement over traditional rule-based chatbots. Rule-based systems often struggle with the nuances of natural language and lack contextual understanding. By leveraging pre-trained language models like Sentence-BERT, our model demonstrates a greater ability to understand the underlying meaning of user input and provide more relevant responses based on semantic similarity with real doctor-patient interactions.

## Future Work

To further enhance the capabilities and robustness of our medical chatbot, we plan to explore the following avenues:

* **Fine-tuning with Transformer-based Architectures:** We aim to fine-tune more advanced transformer-based models like MedGPT, which are specifically designed for medical text, to potentially improve the accuracy and sophistication of the chatbot's responses.
* **Deploying a Real-time Interactive Chatbot API:** We envision deploying the chatbot as a real-time API, making it accessible for integration into various healthcare applications and platforms.
* **Expanding Datasets:** We plan to incorporate more comprehensive datasets that include a wider range of medical conditions, rare diseases, and a broader spectrum of patient demographics to improve the model's generalizability and accuracy.
* **Optimizing Survival Prediction Models:** We intend to explore and optimize survival prediction models using deep learning techniques, such as recurrent neural networks (RNNs) and transformer-based architectures, to gain deeper insights from the integrated medical datasets.

## Summary

This project successfully developed an AI-powered medical chatbot capable of predicting potential diagnoses based on user-provided symptom descriptions. Key achievements include:

* Developed an NLP-based chatbot that predicts diagnoses based on symptom descriptions using machine learning models.
* Achieved high predictive accuracy in chatbot responses by leveraging Sentence-BERT for semantic similarity matching.
* Implemented a user-friendly interface via Streamlit for public accessibility and ease of interaction.
