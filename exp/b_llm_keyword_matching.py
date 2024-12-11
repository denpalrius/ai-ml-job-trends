import os
import sys
import traceback
import time
import re
from dotenv import load_dotenv
import json
from functools import wraps
import os
from datetime import datetime
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# THIS IS A SCRIPT TO AUTO-GENERATE THE LLM CLASSIFICATION FOR JOB POSTINGS USING GEMINI API
# IT IS NOT MEANT TO BE RUN OUTSIDE OF THE JUPYTER NOTEBOOK ENVIRONMENT

import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt, 
    wait_exponential,
    retry_if_exception_type
)
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

filename = "data/job_postings_normalized.parquet"
job_postings = pd.read_parquet(filename)

print(f"{len(job_postings):,} job postings loaded from {filename}")
print(job_postings.sample(5))


class QuotaTracker:
    def __init__(self, rpm_limit=14, daily_limit=1400):
        self.rpm_limit = rpm_limit
        self.daily_limit = daily_limit
        self.state_file = "data/llm/config/quota_state.json"
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        self.load_state()
        
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                # Convert ISO strings to timestamps
                self.minute_calls = [
                    datetime.fromisoformat(ts).timestamp() 
                    for ts in state.get('minute_calls', [])
                ]
                self.daily_calls = [
                    datetime.fromisoformat(ts).timestamp() 
                    for ts in state.get('daily_calls', [])
                ]
        else:
            self.minute_calls = []
            self.daily_calls = []
    
    def save_state(self):
        state = {
            'minute_calls': [
                datetime.fromtimestamp(ts).isoformat() 
                for ts in self.minute_calls
            ],
            'daily_calls': [
                datetime.fromtimestamp(ts).isoformat() 
                for ts in self.daily_calls
            ]
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
    
    def check_quota(self):
        now = datetime.now().timestamp()
        
        # Clean old timestamps
        minute_ago = now - 60
        day_ago = now - 86400
        
        self.minute_calls = [ts for ts in self.minute_calls if ts > minute_ago]
        self.daily_calls = [ts for ts in self.daily_calls if ts > day_ago]
        
        if len(self.minute_calls) >= self.rpm_limit:
            wait_time = 61 - (now - min(self.minute_calls))
            print(f"Rate limit: {len(self.minute_calls)}/{self.rpm_limit} requests this minute")
            return False, int(wait_time)
        
        if len(self.daily_calls) >= self.daily_limit:
            wait_time = 86400 - (now - min(self.daily_calls))
            print(f"Daily limit: {len(self.daily_calls)}/{self.daily_limit} requests today")
            return False, int(wait_time)
            
        return True, 0
    
    def add_call(self):
        now = datetime.now().timestamp()
        self.minute_calls.append(now)
        self.daily_calls.append(now)
        self.save_state()


def rate_limited(func):
    quota = QuotaTracker()
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        can_proceed, wait_time = quota.check_quota()
        
        if not can_proceed:
            print(f"Rate limit reached. Waiting {wait_time} seconds")
            time.sleep(wait_time)
        
        result = func(*args, **kwargs)
        quota.add_call()
        return result
        
    return wrapper

def get_llm_model():
    gemini_api_key = os.getenv('GEMINI_API_KEY')

    if not gemini_api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable")

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    print(f"LLM model loaded successfully: `{model.model_name}`")

    return model

llm = get_llm_model()

def construct_prompt(job_description):
    return f"""
        Does this job description indicate skills related to Artificial Intelligence, Machine Learning, 
        or Data Science, including techniques like deep learning, computer vision, natural language processing,
        predictive modeling, or data analysis, and tools/frameworks such as TensorFlow, PyTorch, or scikit-learn? 
        
        Job Description: {job_description}
        
        Return response in this exact JSON format:
        {{
            "likelihood": <number 0-10>,
            "reason": "<concise explanation>"
        }}
        """

@rate_limited
def classify_with_gemini(model, job_description):
    try:
        prompt = construct_prompt(job_description)
        response = model.generate_content(prompt)

        json_match = re.search(r"```json\n(.*?)\n```", response.text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
            return result["likelihood"], result["reason"]
        else:
            print(f"Could not find JSON in response: {response.text}")
            return -1, "Failed to extract JSON from response"
        
    except ResourceExhausted as e:
        print(f"Hit API quota limit: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return -1, str(e)

def get_chunk_files(checkpoint_dir):
    import glob
    chunk_files = glob.glob(f"{checkpoint_dir}/chunk_*.parquet")
    return sorted(chunk_files)

def get_next_chunk_number(checkpoint_dir):
    chunk_files = get_chunk_files(checkpoint_dir)
    if not chunk_files:
        return 0
    last_chunk = chunk_files[-1]
    return int(last_chunk.split('_')[-1].split('.')[0]) + 1

def classify_jobs_with_llm(df, model, checkpoint_dir, batch_size):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    chunk_number = get_next_chunk_number(checkpoint_dir)
    start_index = chunk_number * batch_size
    
    if start_index >= len(df):
        print("All chunks processed")
        return pd.concat([pd.read_parquet(f) for f in get_chunk_files(checkpoint_dir)])
        
    print(f"Starting from chunk {chunk_number} (index {start_index})")
    
    try:
        for i in range(start_index, len(df), batch_size):
            batch = df.iloc[i:i+batch_size].copy()
            batch['gemini_likelihood'] = 0
            batch['gemini_reason'] = ""
            
            for index, row in batch.iterrows():
                skills_str = ", ".join(row['job_skills'])
                job_desc = f"{row['job_title']} {row['search_position']} {row['company']} {skills_str}"
                likelihood, reason = classify_with_gemini(model, job_desc)
                
                batch.loc[index, 'gemini_likelihood'] = likelihood
                batch.loc[index, 'gemini_reason'] = reason
                
            chunk_file = f"{checkpoint_dir}/chunk_{chunk_number:04d}.parquet"
            batch.to_parquet(chunk_file)
            print(f"Saved chunk {chunk_number}")
            chunk_number += 1
            
        combined_chunk_files = pd.concat([pd.read_parquet(f) for f in get_chunk_files(checkpoint_dir)])
        print(f"Processed {len(combined_chunk_files):,} jobs")
        
        return pd.concat([pd.read_parquet(f) for f in get_chunk_files(checkpoint_dir)])
        
    except Exception as e:
        print(f"Error processing chunk {chunk_number}: {e}")
        return None

checkpoint_dir = "data/llm/chunks"
os.makedirs(checkpoint_dir, exist_ok=True)

classify_jobs_with_llm(
    job_postings,
    llm,
    checkpoint_dir=checkpoint_dir,
    batch_size=10,
)

print("All job postings classified with LLM")


def load_processed_chunks(checkpoint_dir='data/llm/chunks'):
    try:
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        chunk_files = sorted(
            glob.glob(f"{checkpoint_dir}/chunk_*.parquet"),
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        
        if not chunk_files:
            raise FileNotFoundError("No chunk files found")
            
        chunks = []
        for file in chunk_files:
            chunk = pd.read_parquet(file)
            chunks.append(chunk)
            
        combined_df = pd.concat(chunks, axis=0)
        print(f"Loaded {len(chunk_files)} chunks with {len(combined_df)} total records")
        
        return combined_df
        
    except Exception as e:
        print(f"Error loading chunks: {e}")
        return None

processed_chubks = load_processed_chunks(checkpoint_dir=checkpoint_dir)

print(processed_chubks.sample(5))

processed_chubks[processed_chubks['gemini_likelihood'] > 0]['gemini_likelihood'].value_counts().sort_index()

likelihood_distribution = (
    processed_chubks[processed_chubks["gemini_likelihood"] > 0]["gemini_likelihood"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 6))
likelihood_distribution.plot(kind="bar", color="blue", alpha=0.7)
plt.title("Distribution of Keyword Likelihood Count (> 0)")
plt.xlabel("Keyword Likelihood (LLM Classified Jobs)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
