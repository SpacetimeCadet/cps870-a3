import requests
import json
import pandas as pd

# Configuration
API_KEY = "template"
MODEL = "openai/gpt-3.5-turbo"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load CSV data
exec_data = pd.read_csv("nlp/new/actually_good.csv")  # First CSV
turnover_data = pd.read_csv("CHECKME.csv")  # Second CSV
stock_data = pd.read_csv("nlp/new/turnover.csv")  # Third CSV

def query_executive(name=None, company=None, year=None):
    """Query executive information from the first CSV"""
    df = exec_data.copy()
    if name:
        df = df[df['name'].str.contains(name, case=False)]
    if company:
        df = df[df['conm'].str.contains(company, case=False)]
    if year:
        df = df[(df['start_year'] <= int(year)) & (df['end_year'] >= int(year))]
    return df.to_dict('records')

def query_sentiment(name=None, company=None, before_after=None):
    """Query sentiment data from the second CSV"""
    df = turnover_data.copy()
    if name:
        df = df[df['name'].str.contains(name, case=False)]
    if company:
        df = df[df['company'].str.contains(company, case=False)]
    
    # Filter based on the before/after query
    if before_after == 'before':
        df = df[df['end_year'] < df['start_year'].max()]  # Sentiment before turnover
    elif before_after == 'after':
        df = df[df['start_year'] > df['start_year'].min()]  # Sentiment after turnover

    return df.to_dict('records')

def query_stock_performance(name=None, company=None, year=None):
    """Query stock performance from the third CSV"""
    df = stock_data.copy()
    if name:
        df = df[df['exec_name'].str.contains(name, case=False)]
    if company:
        df = df[df['company'].str.contains(company, case=False)]
    if year:
        df = df[df['year'] == int(year)]
    return df.to_dict('records')

def get_avg_sentiment_score(sentiment_data):
    """Helper function to calculate average sentiment score"""
    if sentiment_data:
        scores = [article['sentiment_score'] for article in sentiment_data]
        return sum(scores) / len(scores)
    return None

def chat_with_bot():
    print("Executive Information Bot (type 'quit' to exit)")
    messages = []
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # Check if the question is about executives
        if any(keyword in user_input.lower() for keyword in ['ceo', 'executive', 'sentiment', 'stock', 'performance']):
            try:
                # Extract entities from question
                company = None
                year = None
                name = None
                before_after = None  # To track whether the query is 'before' or 'after' turnover
                
                # Try to extract company and year dynamically
                company_candidates = [comp for comp in exec_data['conm'].unique() if comp.lower() in user_input.lower()]
                if company_candidates:
                    company = company_candidates[0]
                
                # Extract year
                for word in user_input.split():
                    if word.isdigit() and len(word) == 4:
                        year = word
                
                # Check if the user is asking for sentiment 'before' or 'after'
                if "before" in user_input.lower():
                    before_after = 'before'
                elif "after" in user_input.lower():
                    before_after = 'after'
                
                # Query data
                exec_info = query_executive(name, company, year)
                sentiment_info = query_sentiment(name, company, before_after)
                stock_info = query_stock_performance(name, company, year)
                
                # Format context for LLM
                context = {
                    "executive_info": exec_info,
                    "sentiment_info": sentiment_info,
                    "stock_info": stock_info
                }
                
                # Handle sentiment score request
                if 'sentiment' in user_input.lower():
                    avg_sentiment = get_avg_sentiment_score(sentiment_info)
                    if avg_sentiment is not None:
                        bot_reply = f"The average sentiment score for {company} during the {before_after} period is: {avg_sentiment:.4f}"
                    else:
                        bot_reply = "No sentiment data found for the specified period."
                # Handle stock or volatility queries
                elif 'stock' in user_input.lower() or 'volatility' in user_input.lower():
                    if stock_info:
                        stock_reply = []
                        for stock in stock_info:
                            stock_reply.append(f"Year: {stock['year']}, Avg Close Before: {stock['avg_close_before']}, "
                                                f"Avg Close After: {stock['avg_close_after']}, Price Change: {stock['price_change']}, "
                                                f"Volatility Before: {stock['volatility_before']}, Volatility After: {stock['volatility_after']}")
                        bot_reply = "\n".join(stock_reply)
                    else:
                        bot_reply = f"No stock performance data found for {company} in {year}."
                else:
                    # Use the bot for general queries
                    system_msg = {
                        "role": "system",
                        "content": f"You are a helpful assistant that answers questions about corporate executives. Here is relevant data: {json.dumps(context)}"
                    }
                    enhanced_messages = [system_msg] + messages[-4:]  # Keep last few messages
                    
                    headers = {
                        "Authorization": f"Bearer {API_KEY}",
                        "Content-Type": "application/json"
                    }
                    
                    data = {
                        "model": MODEL,
                        "messages": enhanced_messages,
                        "temperature": 0.3  # Lower temp for factual answers
                    }
                    
                    try:
                        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
                        response_data = response.json()
                        
                        if response.status_code == 200:
                            bot_reply = response_data['choices'][0]['message']['content']
                            print("Bot:", bot_reply)
                            messages.append({"role": "assistant", "content": bot_reply})
                        else:
                            print("Error:", response_data.get('error', 'Unknown error occurred'))
                    except Exception as e:
                        print("An error occurred:", str(e))
            
            except Exception as e:
                print(f"Error querying data: {str(e)}")
                enhanced_messages = messages
        else:
            enhanced_messages = messages

if __name__ == "__main__":
    chat_with_bot()
