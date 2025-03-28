import requests
import json
import pandas as pd
import re

# Configuration
API_KEY = ""  # Add your API key here
MODEL = "openai/gpt-3.5-turbo"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Load CSV data
exec_data = pd.read_csv("nlp/new/actually_good.csv")
turnover_data = pd.read_csv("CHECKME.csv") # this is actually sentiment cssv
stock_data = pd.read_csv("nlp/new/turnover.csv")

class ConversationState:
    def __init__(self):
        self.awaiting_sentiment_period = False
        self.current_context = {}
        self.current_executive = None
        
# Initialize state globally
state = ConversationState()

def extract_entities(text):
    """Extract name, company, and year from user input"""
    entities = {'name': None, 'company': None, 'year': None}
    
    # Extract year
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    if year_match:
        entities['year'] = year_match.group()
    
    # Extract name first (more reliable for name-only queries)
    name_match = re.search(r'(?:mr\.?|ms\.?|mrs\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text, re.I)
    if name_match:
        entities['name'] = name_match.group(1).strip()
    
    # Then extract company if present
    text_lower = text.lower()
    best_match = None
    max_length = 0
    
    for company in set(exec_data['conm'].dropna().unique()):
        if not isinstance(company, str):
            continue
            
        company_lower = company.lower()
        if company_lower in text_lower:
            if len(company_lower) > max_length:
                best_match = company
                max_length = len(company_lower)
    
    entities['company'] = best_match
    
    return entities

def query_executive(name=None, company=None, year=None):
    df = exec_data.copy()
    
    # Strict company matching
    if company:
        # First try exact match (case insensitive)
        exact_match = (df['conm'].str.lower() == company.lower()) | (df['tic'].str.lower() == company.lower())
        
        if exact_match.any():
            df = df[exact_match]
        else:
            # If no exact match, try contains but be more strict
            company_matches = (
                df['conm'].str.lower().str.contains(r'\b' + re.escape(company.lower()) + r'\b', na=False) |
                df['tic'].str.lower().str.contains(r'\b' + re.escape(company.lower()) + r'\b', na=False)
            )
            df = df[company_matches]
    
    # Rest remains the same...
    if name:
        df = df[df['name'].str.contains(re.escape(name), case=False, na=False)]
    
    if year:
        try:
            year = int(year)
            df = df[(df['start_year'] <= year) & (year <= df['end_year'])]
        except:
            pass
    
    if df.empty:
        return None
    
    df['match_score'] = (
        df['name'].notna().astype(int) * 3 +
        df['conm'].notna().astype(int) * 2 +
        df['tic'].notna().astype(int) * 2 +
        df['start_year'].notna().astype(int)
    )
    
    return df.sort_values('match_score', ascending=False).head(1).to_dict('records')[0]

def query_sentiment(name=None, company=None, period=None):
    df = turnover_data.copy()
    if name:
        df = df[df['name'].str.contains(re.escape(name), case=False, na=False)]
    if company:
        df = df[df['company'].str.contains(re.escape(company), case=False, na=False)]
    if period:
        period_flag = 1 if period == 'after' else 0
        df = df[df['turnover_before'] == period_flag]
    
    if df.empty:
        return None
    
    avg_score = df['sentiment_score'].mean()
    article_count = len(df)
    sample_articles = df[['title', 'sentiment_score']].to_dict('records')
    
    return {
        'average_score': avg_score,
        'article_count': article_count,
        'period': period,
        'sample_articles': sample_articles
    }

def build_context():
    context = []
    
    if state.current_executive:
        exec = state.current_executive
        context.append(f"Executive: {exec['name']} at {exec.get('conm', exec.get('company', 'unknown company'))}")
    
    if state.current_context.get('sentiment'):
        sent = state.current_context['sentiment']
        if sent:
            period_desc = "after" if sent['period'] == 'after' else "before"
            context.append(f"Sentiment {period_desc} tenure: {sent['average_score']:.2f} (from {sent['article_count']} articles)")
    
    return " | ".join(context) if context else "No specific context"

def handle_response(user_input):
    global state
    # Reset context if not a follow-up question
    if not any(q in user_input.lower() for q in ['what about', 'how about']):
        state.current_context = {}
    if is_factual_query(user_input):
        csv_response = try_answer_from_csv(user_input)
        if csv_response:
            return {
                'response': csv_response,
                'source': 'CSV'
            }
    
    # If not found in CSV or not a simple factual query, use LLM
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_message = {
        "role": "system",
        "content": f"""You are an executive data analyst. Current context: {build_context()}
        When discussing sentiment:
        - Scores range from -1 (negative) to 1 (positive)
        - Near 0 is neutral
        - Provide specific examples when available"""
    }
    
    messages = [system_message, {"role": "user", "content": user_input}]
    
    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return {
            'response': response.json()['choices'][0]['message']['content'],
            'source': 'LLM'
        }
    except Exception as e:
        return {
            'response': f"Error processing request: {str(e)}",
            'source': 'System'
        }

def is_factual_query(text):
    """Determine if the query is a simple factual question that might be in our CSV"""
    factual_keywords = [
        'who was', 'who is', 'ceo of', 'ceo in', 
        'executive of', 'executive in', 'leader of'
    ]
    return any(kw in text.lower() for kw in factual_keywords)

def try_answer_from_csv(text):
    """Attempt to answer the question from CSV data"""
    entities = extract_entities(text)
    
    # If we have a name but no company, prioritize name matches
    if entities['name'] and not entities['company']:
        # Search for executives with matching names
        name_matches = exec_data[exec_data['name'].str.contains(entities['name'], case=False, na=False)]
        
        if not name_matches.empty:
            # If we have multiple matches, return the most recent one
            best_match = name_matches.sort_values('end_year', ascending=False).iloc[0]
            
            # Format the response
            company = best_match.get('conm', best_match.get('company', 'unknown company'))
            period = f"{int(best_match['start_year'])}-{int(best_match['end_year'])}" if pd.notna(best_match['start_year']) else "unknown period"
            return f"{best_match['name']} was CEO of {company} ({period})"
    
    # Original company/year logic for other cases
    if not entities['company'] and not entities['year']:
        return None
    
    exec_info = query_executive(
        company=entities.get('company'),
        year=entities.get('year')
    )
    
    if not exec_info:
        return None
    
    # Format the response
    company = exec_info.get('conm', exec_info.get('company', 'unknown company'))
    year = entities.get('year', 'the specified time')
    return f"In {year}, the CEO of {company} was {exec_info['name']}."

def handle_sentiment_query(period):
    if not state.current_executive:
        return {
            'response': "I need to know which executive you're asking about first.",
            'source': 'System'
        }
    
    sentiment = query_sentiment(
        name=state.current_executive.get('name'),
        company=state.current_executive.get('conm', state.current_executive.get('company')),
        period=period
    )
    
    state.current_context['sentiment'] = sentiment
    state.awaiting_sentiment_period = False
    
    if not sentiment:
        return {
            'response': f"No sentiment data found for {state.current_executive.get('name')} {period} tenure.",
            'source': 'System'
        }
    
    response = []
    response.append(f"[CSV Data] Average sentiment {period} tenure: {sentiment['average_score']:.2f} (from {sentiment['article_count']} articles)")
    
    if sentiment['average_score'] > 0.3:
        response.append("[LLM Interpretation] This indicates generally positive sentiment.")
    elif sentiment['average_score'] < -0.3:
        response.append("[LLM Interpretation] This indicates generally negative sentiment.")
    else:
        response.append("[LLM Interpretation] This indicates neutral or mixed sentiment.")
    
    if sentiment.get('sample_articles'):
        examples = sentiment['sample_articles'][:2]
        response.append("\n[CSV Data] Example articles:")
        for article in examples:
            score = article['sentiment_score']
            sentiment_label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
            response.append(f"- '{article['title']}' ({sentiment_label}, score: {score:.2f})")
    
    return {
        'response': "\n".join(response),
        'source': 'CSV+LLM'
    }

def chat_with_bot():
    print("Executive Data Analyst Bot (type 'quit' to exit)")
    global state
    
    while True:
        try:
            if state.awaiting_sentiment_period:
                user_input = input("You: ").strip().lower()
                if user_input in ['before', 'after']:
                    result = handle_sentiment_query(user_input)
                    print(f"Bot: {result['response']}")
                    print(f"[Source: {result['source']}]")
                    continue
                else:
                    print("Bot: Please specify 'before' or 'after'")
                    print("[Source: System]")
                    continue
                
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                break
                
            # Don't reset context for simple greetings
            if user_input.lower() not in ['hi', 'hello', 'hey']:
                # Only reset context if we're not in a follow-up question
                if not any(q in user_input.lower() for q in ['what about', 'how about']):
                    state.current_context = {}
            
            entities = extract_entities(user_input)
            exec_info = query_executive(
                name=entities.get('name'),
                company=entities.get('company'),
                year=entities.get('year')
            )
            
            if exec_info:
                # Verify extracted name matches queried name
                if entities.get('name') and (entities['name'].lower() not in exec_info['name'].lower()):
                    print(f"Bot: Found {exec_info['name']} matching your company query.")
                else:
                    state.current_executive = exec_info
            
            if 'sentiment' in user_input.lower():
                if not any(kw in user_input.lower() for kw in ['before', 'after']):
                    if not state.current_executive:
                        print("Bot: I need to know which executive you're asking about first.")
                        print("[Source: System]")
                        continue
                    print(f"Bot: Would you like {state.current_executive.get('name')}'s sentiment right before their departure or after?")
                    print("[Source: System]")
                    state.awaiting_sentiment_period = True
                    continue
            
            if user_input.lower() in ['what about after?', 'what about before?']:
                period = 'after' if 'after' in user_input.lower() else 'before'
                result = handle_sentiment_query(period)
                print(f"Bot: {result['response']}")
                print(f"[Source: {result['source']}]")
                continue
            
            result = handle_response(user_input)
            print(f"Bot: {result['response']}")
            print(f"[Source: {result['source']}]")
            
        except Exception as e:
            print(f"Bot: Error processing your request - {str(e)}")
            print("[Source: System]")
            continue

if __name__ == "__main__":
    chat_with_bot()
