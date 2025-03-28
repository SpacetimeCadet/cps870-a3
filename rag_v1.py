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
from flask import Flask, render_template, request
from markupsafe import Markup

# Initialize User Interface
app = Flask (__name__)

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

llm = Mistral(api_key = api_key)
chat_model = "mistral-large-latest"
embedding_model = "mistral-embed"
fact_file_path = ('./data/perf_sent.csv')
articles_file_path = ('../CHECKME3.csv')

df = pd.read_csv(fact_file_path)
# exec_name,company,end_year,ticker,
# avg_close_before,avg_close_after,price_change,percent_change,volatility_before,volatility_after,volatility_change,
# avg_close_before_3mo,avg_close_after_3mo,price_change_3mo,percent_change_3mo,volatility_before_3mo,volatility_after_3mo,volatility_change_3mo,
# avg_close_before_1mo,avg_close_after_1mo,price_change_1mo,percent_change_1mo,volatility_before_1mo,volatility_after_1mo, volatility_change_1mo,
# tic,execid,start_year,end_month,tenure_len, sentiment_before, sentiment_after

df2 = pd.read_csv(articles_file_path)
df2 = df2[df2['sentiment_score'] != 0.0]
# loader = CSVLoader(file_path = articles_file_path)
# article_texts = loader.load_and_split() # each row

def verbalize(row):
    return (
        f"Executive {row['exec_name']} left {row['ticker']} {row['company']} in {row['end_month']}/{row['end_year']}.\n"
        f"{row['exec_name']} had a tenure length of {row['tenure_len']} years starting from {row['start_year']} to {row['end_year']}. \n"
        
        f"The average closing price over the 12 months before departure was ${row['avg_close_before']}. \n"
        f"In the 12 months after, the average closing price was ${row['avg_close_after']}. \n"
        f"The stock price changed by ${row['price_change']} over that period. \n"
        f"This represents a {row['percent_change']}% {'increase' if row['percent_change'] >= 0 else 'decrease'} in price. \n\n"

        f"Volatility in the 12 months before departure was {row['volatility_before']}. \n"
        f"Volatility 12 months after the departure was {row['volatility_after']}. \n"
        f"This reflects a change in volatility of {row['volatility_change']}. \n"

        f"The average closing price over the 3 months before departure was ${row['avg_close_before_3mo']}. \n"
        f"In the 3 months after, the average closing price was ${row['avg_close_after_3mo']}. \n"
        f"The stock price changed by ${row['price_change_3mo']} over that period. \n"
        f"This represents a {row['percent_change_3mo']}% {'increase' if row['percent_change_3mo'] >= 0 else 'decrease'} in price. \n"

        f"Volatility in the 3 months before departure was {row['volatility_before_3mo']}. \n"
        f"Volatility 3 months after the departure was {row['volatility_after_3mo']}. \n"
        f"This reflects a change in volatility of {row['volatility_change_3mo']}. \n\n"

        f"The average closing price over the 1 month before departure was ${row['avg_close_before_1mo']}. \n"
        f"In the 1 month after, the average closing price was ${row['avg_close_after_1mo']}. \n"
        f"The stock price changed by ${row['price_change_1mo']} over that period. \n"
        f"This represents a {row['percent_change_1mo']}% {'increase' if row['percent_change_1mo'] >= 0 else 'decrease'} in price. \n\n"

        f"Volatility in the 1 months before departure was {row['volatility_before_1mo']}. \n"
        f"Volatility 1 month after the departure was {row['volatility_after_1mo']}. \n"
        f"This reflects a change in volatility of {row['volatility_change_1mo']}. \n"

        f"Average sentiment before the departure was {row['sentiment_before']}. \n"
        f"After the departure, average sentiment was {row['sentiment_after']}. \n"
        f"This represents a {'increase' if (row['sentiment_after'] - row['sentiment_before']) >= 0 else 'decrease'} in sentiment. \n\n"
    )

# chunk articles longer than 8k token limit
def verbalize_article_chunks(row, chunk_len=32000, overlap=200):
    content = row['content']
    chunks = []
    for i in range(0, len(content), chunk_len - overlap):
        subtext = content[i:i+chunk_len]
        chunk = (
            f"This is a chunk of an article written {'before' if row['turnover_before'] == 0 else 'after'} "
            f"the turnover of {row['name']} at {row['company']} in {row['end_year']}, "
            f"with a sentiment score of {row['sentiment_score']:.2f}. {subtext}\n\n"
        )
        chunks.append(chunk)
    return chunks

def get_embeddings_by_chunks(data, chunk_size):
    chunks = [data[x : x + chunk_size] for x in range(0, len(data), chunk_size)]
    embeddings_response = []
    for c in chunks:
        response = llm.embeddings.create(model=embedding_model, inputs=c)
        embeddings_response.append(response)
        time.sleep(1) 
    return [d.embedding for e in embeddings_response for d in e.data]

# query types
# what was sentiment before X	                facts only
# why was sentiment low after X	                articles only
# how did the media react	                    articles only
# how did stock and sentiment shift after X     facts only
# what happened after Y left?	                facts + articles

def classify_query(query):
    explanatory_keywords = [
        'why', 'reason', 'explain', 'because', 
        'how come', 'cause', 'due to', 'factor',
        'led to', 'resulted in', 'what led', 
        'driving', 'underlying', 'behind'
    ]
    if any(w in query.lower() for w in explanatory_keywords): # ["why", "reason", "explain", "because", "media said", "reacted"]):
        return "articles"
    elif any(w in query.lower() for w in ["score", "how much", "percent", "change", "value", "volatility", "stock"]):
        return "facts"
    else:
        return "facts"

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

article_text_chunks = []
for _, row in df2.iterrows():
    for chunk in verbalize_article_chunks(row):
        article_text_chunks.append(chunk)

embeddings = np.array(get_embeddings_by_chunks(texts, 100))
art_embeddings = np.array(get_embeddings_by_chunks(article_text_chunks, 20))


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["GET", "POST"])
def chatbot_response():
    message = request.form["message"]
    query = message
    
    if query.lower() == 'help':
        ceos = get_random_ceos(3)
        ceo_list = "\n".join([f"- {ceo[0]} ({ceo[1]})" for ceo in ceos])
        return f"Here are a few CEOs and their companies to get started:\n{ceo_list}"
    
    prompt = "Using primarily the following information, please tell me "
    
    if classify_query(query) == 'articles':
        retrieved_texts = retrieve(article_text_chunks, art_embeddings, query)
    else:
        retrieved_texts = retrieve(texts, embeddings, query)
    
    # print(f'retrieved texts: {retrieved_texts}\n')
    input_text = prompt + query + "\n" + "\n".join(retrieved_texts[:3])
    chat_response = llm.chat.complete(
        model = chat_model,
        messages = [
            {
                "role": "user",
                "content": f"{input_text}",
            },
        ]
    )

    #print(chat_response.choices[0].message.content)
    response_text = str(chat_response.choices[0].message.content)
    return Markup(response_text.replace('\n', '<br>'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, port=5001)
    
