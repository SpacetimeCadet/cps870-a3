import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from mistralai import Mistral
import pandas as pd
import numpy as np
import faiss
import time
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

llm = Mistral(api_key = api_key)
chat_model = "mistral-large-latest"
embedding_model = "mistral-embed"
file_path = ('./data/perf_data.csv')

df = pd.read_csv(file_path)
# exec_name,company,end_year,ticker,
# avg_close_before,avg_close_after,price_change,percent_change,volatility_before,volatility_after,volatility_change,
# avg_close_before_3mo,avg_close_after_3mo,price_change_3mo,percent_change_3mo,volatility_before_3mo,volatility_after_3mo,volatility_change_3mo,
# avg_close_before_1mo,avg_close_after_1mo,price_change_1mo,percent_change_1mo,volatility_before_1mo,volatility_after_1mo, volatility_change_1mo,
# tic,execid,start_year,end_month,tenure_len

def verbalize(row):
    return (
        f"Executive {row['exec_name']} left {row['ticker']} {row['company']} in {row['end_month']}/{row['end_year']}. "
        f"{row['exec_name']} had a tenure length of {row['tenure_len']} years starting from {row['start_year']} to {row['end_year']}. "
        
        f"The average closing price over the 12 months before departure was ${row['avg_close_before']}."
        f"In the 12 months after, the average closing price was ${row['avg_close_after']}."
        f"The stock price changed by ${row['price_change']} over that period."
        f"This represents a {row['percent_change']}% {'increase' if row['percent_change'] >= 0 else 'decrease'} in price."

        f"Volatility in the 12 months before departure was {row['volatility_before']}."
        f"Volatility 12 months after the departure was {row['volatility_after']}."
        f"This reflects a change in volatility of {row['volatility_change']}."

        f"The average closing price over the 3 months before departure was ${row['avg_close_before_3mo']}."
        f"In the 3 months after, the average closing price was ${row['avg_close_after_3mo']}."
        f"The stock price changed by ${row['price_change_3mo']} over that period."
        f"This represents a {row['percent_change_3mo']}% {'increase' if row['percent_change_3mo'] >= 0 else 'decrease'} in price."

        f"Volatility in the 3 months before departure was {row['volatility_before_3mo']}."
        f"Volatility 3 months after the departure was {row['volatility_after_3mo']}."
        f"This reflects a change in volatility of {row['volatility_change_3mo']}."

        f"The average closing price over the 1 month before departure was ${row['avg_close_before_1mo']}."
        f"In the 1 month after, the average closing price was ${row['avg_close_after_1mo']}."
        f"The stock price changed by ${row['price_change_1mo']} over that period."
        f"This represents a {row['percent_change_1mo']}% {'increase' if row['percent_change_1mo'] >= 0 else 'decrease'} in price."

        f"Volatility in the 1 months before departure was {row['volatility_before_1mo']}."
        f"Volatility 1 month after the departure was {row['volatility_after_1mo']}."
        f"This reflects a change in volatility of {row['volatility_change_1mo']}."
    )

def get_embeddings_by_chunks(data, chunk_size):
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = []
    for c in chunks:
        response = llm.embeddings.create(model=embedding_model, inputs=c)
        embeddings_response.append(response)
        time.sleep(1) 
    return [d.embedding for e in embeddings_response for d in e.data]

def retrieve(texts, embeddings, query, k=7):
    # Build the FAISS index
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Query embedding
    query_embedding = llm.embeddings.create(
        model=embedding_model,
        inputs=query,
    )
    time.sleep(1)
    query_embedding = np.array(query_embedding.data[0].embedding).reshape(1, -1)
    # Search in FAISS
    _, indices = index.search(query_embedding, k)
    # print(indices)
    retrieved_docs = [texts[i] for i in indices[0]]
    return retrieved_docs




texts = df.apply(verbalize, axis=1).tolist()

# loader = CSVLoader(file_path = file_path)
# docs = loader.load_and_split() # each row

# embeddings = llm.embeddings.create(
#     model=embedding_model,
#     inputs=texts,
# )
embeddings = np.array(get_embeddings_by_chunks(texts, 100))
# query = "Which executive left American Airlines most recently"
# prompt = "Given the following information, please tell me "
# retrieved_texts = retrieve(texts, embeddings, query)
# input = prompt + query + "\n" + retrieved_texts[0]
# print(retrieved_texts[0])

# query_embedding = llm.embeddings.create(
#         model=embedding_model,
#         inputs='whats ur fav food',
#     )
# print(type(np.array(query_embedding.data[0].embedding).reshape(1, -1)))

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        break

    query = user_input
    prompt = "Using primarily the following information, please tell me "
    retrieved_texts = retrieve(texts, embeddings, query)
    print(f'retrieved texts: {retrieved_texts}\n')
    input_text = prompt + query + "\n" + retrieved_texts[0]
    chat_response = llm.chat.complete(
        model = chat_model,
        messages = [
            {
                "role": "user",
                "content": f"{input_text}",
            },
        ]
    )

    print(chat_response.choices[0].message.content)