import re
from collections import Counter
import time

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, Markdown

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

import geopandas as gpd

def main():
    filename = "data/b_job_postings_ai_ml_ds.parquet"
    ai_ml_jobs = pd.read_parquet(filename)

    print(f"{len(ai_ml_jobs):,} job postings loaded from {filename}")
    ai_ml_jobs.sample(3)

    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    def combine_row_text(row):
        return f"{row['job_title']} {row['search_position']} {', '.join(row['job_skills'])}".lower()

    def perform_sentiment_analysis():
        def analyze_sentiment_bert(text):
            max_length = 512
            truncated_text = text[:max_length]
            result = sentiment_analyzer(truncated_text)
            sentiment = result[0]["label"], result[0]["score"]
            return sentiment

        def add_sentiment_bert(df):
            df = df.copy()
            combined_text = df.apply(lambda row: combine_row_text(row), axis=1)
            sentiments = []
            for desc in tqdm(combined_text, desc="Analyzing Sentiment"):
                sentiments.append(analyze_sentiment_bert(desc))
            df["sentiment_label"], df["sentiment_score"] = zip(*sentiments)
            
            return df

        start_time = time.time()

        ai_ml_jobs_analysed = add_sentiment_bert(ai_ml_jobs)

        end_time = time.time()
        print(f"Processed {len(ai_ml_jobs_analysed):,} job postings in {end_time - start_time:.0f} seconds")

        filename_analysed = "data/c_job_postings_ai_ml_sent_analysed.parquet"
        ai_ml_jobs_analysed.to_parquet(filename_analysed)

        ai_ml_jobs_analysed.head()

        plt.figure(figsize=(10, 6))
        sns.countplot(data=ai_ml_jobs_analysed, x='sentiment_label')
        plt.title('Distribution of Sentiment Labels')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(data=ai_ml_jobs_analysed, x='sentiment_score', bins=20, kde=True)
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.show()

        sentiment_by_title = (
            ai_ml_jobs_analysed.groupby("job_title")["sentiment_score"]
            .mean()
            .sort_values(ascending=False)
        )
        print(sentiment_by_title[:10])

        plt.figure(figsize=(10, 6))
        sentiment_by_title.plot(kind="bar")
        plt.title("Average Sentiment Score by Job Title")
        plt.xlabel("Job Title")
        plt.ylabel("Average Sentiment Score")
        plt.show()

        sns.scatterplot(
            x="sentiment_score",
            y="keyword_likelihood",
            hue="job_level",
            data=ai_ml_jobs_analysed,
        )

        plt.show()

        ai_ml_jobs_analysed[
            (ai_ml_jobs_analysed["keyword_likelihood"] > 5)
            & (ai_ml_jobs_analysed["sentiment_score"] > 0.5)
        ]

        top_cities = ai_ml_jobs_analysed['search_city'].value_counts().head(10).index
        city_sentiment = ai_ml_jobs_analysed[ai_ml_jobs_analysed['search_city'].isin(top_cities)].groupby('search_city')['sentiment_score'].mean()

        city_sentiment.plot(kind='bar')
        plt.xlabel('City')
        plt.ylabel('Average Sentiment Score')
        plt.title('Top 10 Cities with AI/ML Job Postings and Their Average Sentiment Score')
        plt.xticks(rotation=45)
        plt.show()

    # perform_sentiment_analysis()
    
    def perform_skill_analysis():
        ai_ml_terms = """
        artificial intelligence, machine learning, deep learning, neural networks, computer vision, natural language processing, large language model, reinforcement learning, supervised learning, unsupervised learning, semi-supervised learning, transfer learning, predictive modeling, classification, regression, clustering, convolutional neural networks, cnn, rnn, long short-term memory, lstm, gan, generative adversarial networks, support vector machine, svm, random forests, decision trees, ensemble learning, feature engineering, feature selection, data preprocessing, data mining, big data, data science, data analysis, data visualization, sentiment analysis, chatbots, speech recognition, image recognition, object detection, time series analysis, recommender systems, autonomous systems, robotics, chatbot development, nlp algorithms, tensorFlow, keras, pytorch, scikit-learn, openai, machine learning algorithms, model training, model evaluation
        """

        ai_ml_frameworks = """
        Python, R, Julia, SQL, Apache Spark, Databricks, Jupyter Notebook, Google Cloud AI Platform, Amazon SageMaker, Microsoft Azure Machine Learning, Docker, Kubernetes, Git, GitHub, Anaconda, Weights & Biases, MLflow, Apache Kafka, Airflow, Tableau, Power BI, Looker, MATLAB, Scala, C++, Java, PySpark, Databricks, Snowflake, BigQuery, Vertex AI, OpenCV, NLTK, spaCy, Gensim, Hugging Face Transformers, Ray, Dask, Kedro, DVC (Data Version Control), Great Expectations, Kedro, Weights & Biases, Streamlit, FastAPI, Dash, Flask, XGBoost, LightGBM
        """

        ai_ml_acronyms = """
        RAG, LLM, GenAI, ML, DL, NLP, CV, RL, GAN, CNN, RNN, LSTM, SVM, SAA, AGI, AIoT, HCI, TF, PyTorch, KNN, SVD, BERT, GPT, T5, VAE, BIM, BOM, IoT, CICD, MLOps, AI/ML, API, RPA, ETL, DNN, RNN, BFS, AIaaS, MLaaS, DLaaS, Jupyter, K8s, EDA, RNN, BERT, VQA, CVPR, MT, FL, AI-ML, Keras, XLNet
        """

        ai_terms_list = [item.strip() for item in ai_ml_terms.split(",")]
        ai_ml_frameworks_list = [item.strip() for item in ai_ml_frameworks.split(",")]
        ai_acronyms_list = [item.strip() for item in ai_ml_acronyms.split(",")]

        ai_ml_keywords = set(ai_terms_list + ai_ml_frameworks_list + ai_acronyms_list)

        print(f"Number of AI/ML keywords: {len(ai_ml_keywords)}")

        ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer)

        def extract_skill_context_bert(text, keywords):
            entities = ner_pipeline(text)
            skill_contexts = []

            for entity in entities:
                if entity['word'].lower() in keywords:
                    skill_contexts.append({
                        "skill": entity['word'],
                        "entity_type": entity['entity'],
                        "context": text[max(0, entity['start'] - 30):entity['end'] + 30]
                    })

            return skill_contexts

        def add_skill_context_bert(df, keywords):
            df = df.copy()
            combined_text = df.apply(lambda row: combine_row_text(row), axis=1)
            skill_contexts = []
            for desc in tqdm(combined_text, desc="Extracting Skill Contexts"):
                contexts = extract_skill_context_bert(desc, keywords)
                skill_contexts.append(contexts)
            df['skill_contexts'] = skill_contexts
            return df

        keywords = [keyword.lower() for keyword in ai_ml_keywords]
        ai_ml_jobs_with_contexts = add_skill_context_bert(ai_ml_jobs, keywords)

        print(ai_ml_jobs_with_contexts[['skill_contexts']])

        filename_analysed_ctx = "data/c_job_postings_ai_ml_analysed_ctx.parquet"
        ai_ml_jobs_with_contexts.to_parquet(filename_analysed_ctx)

    perform_skill_analysis()
    

if __name__ == "__main__":
    main()