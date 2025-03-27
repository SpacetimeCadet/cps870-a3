import pandas as pd
import re
import string
from collections import Counter
import numpy as np

# Global Variables
input_file = "input.csv"


class CSVQuestionAnswerer:
    
    def __init__(self, csv_path=input_file):
        # Load CSV
        self.csv_path = csv_path
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns.")
            print(f"Columns: {', '.join(self.df.columns)}")
            
            # Store column types for better query handling
            self.column_types = {col: str(self.df[col].dtype) for col in self.df.columns}
            
            # Store some sample data for understanding the content
            self.sample_data = self.df.head(3)
            
            # Hard-code the column names from input.csv
            # This ensures we're working with the exact structure of the file
            self.csv_columns = {
                'name': 'name',
                'company': 'company',
                'year': 'year',
                'turnover_type': 'turnover_type',
                'sentiment': 'sentiment',
                'turnover_before': 'turnover_before'
            }
            
            # Define mappings between query types and actual column names in the CSV
            self.query_to_column_mapping = {
                'name': 'name',
                'company': 'company',
                'year': 'year',
                'turnover_type': 'turnover_type',
                'sentiment': 'sentiment',
                'turnover_before': 'turnover_before'
            }
            
            # Identify the first row to check if it contains headers
            self.has_header_row = self._check_for_header_row()
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            self.df = None
    
    def _check_for_header_row(self):
        """
        Check if the first row might contain header information rather than data
        
        Returns:
            bool: True if the first row appears to be a header row
        """
        # We know the structure of input.csv, so we can assume it has a header row
        # This function is kept for compatibility with the original design
        return True
    
    def _preprocess_question(self, question):
        """
        Preprocess the question by converting to lowercase and removing punctuation.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            str: Preprocessed question.
        """
        # Convert to lowercase
        question = question.lower()
        
        # Remove punctuation
        question = question.translate(str.maketrans('', '', string.punctuation))
        
        return question
    
    def _extract_keywords(self, question):
        """
        Extract keywords from the question, focusing on column names and value indicators.
        
        Args:
            question (str): The preprocessed question.
            
        Returns:
            list: List of keywords.
        """
        # Remove common stop words
        stop_words = ['the', 'a', 'an', 'in', 'on', 'at', 'is', 'are', 'and', 
                      'to', 'of', 'for', 'with', 'what', 'which', 'where',
                      'when', 'how', 'many', 'much', 'do', 'does', 'list', 'tell',
                      'me', 'show', 'give', 'find', 'all', 'that', 'have', 'has']
        
        words = [word for word in question.split() if word not in stop_words]
        
        # Add column names as keywords
        column_keywords = list(self.csv_columns.values())
        
        # Add value indicators for specific columns
        value_indicators = {
            'turnover_type': ['voluntary', 'involuntary', 'fired', 'quit', 'resigned', 'forced'],
            'sentiment': ['positive', 'negative', 'neutral', 'good', 'bad'],
            'year': [str(y) for y in range(2000, 2025)]  # Common years
        }
        
        value_keywords = []
        for column, indicators in value_indicators.items():
            for indicator in indicators:
                if indicator in question.lower():
                    value_keywords.append(indicator)
        
        # Create a set of keywords
        all_keywords = set(words + column_keywords + value_keywords)
        
        return list(all_keywords)
    
    def _identify_query_type(self, question):
        """
        Identify the type of query based on the question by matching keywords to our known columns.
        
        Args:
            question (str): The preprocessed question.
            
        Returns:
            str: Type of query corresponding to a column in the CSV
        """
        # Mapping of keywords to query types
        keyword_to_query_map = {
            # name keywords
            'who': 'name',
            'ceo': 'name',
            'executive': 'name',
            'person': 'name',
            
            # company keywords
            'company': 'company',
            'business': 'company',
            'corporation': 'company',
            'organization': 'company',
            'firm': 'company',
            
            # year keywords
            'when': 'year',
            'year': 'year',
            'time': 'year',
            'date': 'year',
            
            # turnover_type keywords
            'type': 'turnover_type',
            'voluntary': 'turnover_type',
            'involuntary': 'turnover_type',
            'fired': 'turnover_type',
            'resigned': 'turnover_type',
            'quit': 'turnover_type',
            'forced': 'turnover_type',
            
            # sentiment keywords
            'sentiment': 'sentiment',
            'feeling': 'sentiment',
            'reaction': 'sentiment',
            'positive': 'sentiment',
            'negative': 'sentiment',
            'market': 'sentiment',
            
            # turnover_before keywords
            'previous': 'turnover_before',
            'before': 'turnover_before',
            'turnover': 'turnover_before',
            'history': 'turnover_before',
            'prior': 'turnover_before'
        }
        
        # Split question into words
        words = question.split()
        
        # Check for direct column name mentions
        for column in self.csv_columns.values():
            if column in question:
                return column
        
        # Check for keyword matches
        for word in words:
            if word in keyword_to_query_map:
                return keyword_to_query_map[word]
                
        # If no specific column is identified, default to company
        # This is a reasonable default since most queries might be about companies
        return 'company'
    
    def _extract_entity_name(self, question):
        """
        Extract potential entity names (like CEO or company names) from the question.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            tuple: (primary_entity, list_of_proper_names)
                - primary_entity: Primary entity name (str or None)
                - list_of_proper_names: List of all proper names found (list)
        """
        proper_names = []
        primary_entity = None
        
        # Look for quoted strings which often contain names (highest priority)
        quoted_matches = re.findall(r'"([^"]+)"', question)
        quoted_matches.extend(re.findall(r"'([^']+)'", question))
        
        if quoted_matches:
            primary_entity = quoted_matches[0]
            proper_names.extend(quoted_matches)
        
        # Look for capitalized words that might be names or companies
        # This regex finds sequences of capitalized words
        capitalized_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
        if capitalized_matches:
            proper_names.extend(capitalized_matches)
            # If we don't have a primary entity yet, use the longest match
            if not primary_entity and capitalized_matches:
                primary_entity = max(capitalized_matches, key=len)
        
        # Look for company indicators followed by words
        company_indicators = ['at', 'from', 'of', 'for', 'about']
        for indicator in company_indicators:
            pattern = rf'\b{indicator}\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)\b'
            indicator_matches = re.findall(pattern, question, re.IGNORECASE)
            if indicator_matches:
                proper_names.extend(indicator_matches)
                # If no primary entity yet, use the first indicator match
                if not primary_entity:
                    primary_entity = indicator_matches[0]
        
        # Look for potential special names (common CEO last names or big companies)
        # This list could be expanded with known CEO names and company names
        special_names = ['Apple', 'Google', 'Microsoft', 'Amazon', 'Facebook', 'Tesla', 'Twitter',
                         'Cook', 'Bezos', 'Musk', 'Zuckerberg', 'Gates', 'Jobs', 'Pichai', 'Nadella']
        
        for name in special_names:
            if name.lower() in question.lower():
                # Check if it's a whole word, not part of another word
                pattern = rf'\b{name}\b'
                if re.search(pattern, question, re.IGNORECASE):
                    proper_names.append(name)
                    # If no primary entity yet, use this special name
                    if not primary_entity:
                        primary_entity = name
        
        # If still no entity, look for non-trivial words
        if not primary_entity:
            words = question.split()
            potential_names = [word for word in words if len(word) > 5 and word.isalpha()]
            if potential_names:
                primary_entity = potential_names[0]
        
        # Remove duplicates while preserving order
        unique_proper_names = []
        for name in proper_names:
            if name not in unique_proper_names:
                unique_proper_names.append(name)
        
        return primary_entity, unique_proper_names
    
    def _find_rows_by_query(self, query_type, entity_name=None, proper_names=None):
        """
        Find rows matching the query type and entity names, prioritizing exact proper name matches.
        
        Args:
            query_type (str): Column to query.
            entity_name (str, optional): Primary entity name to search for.
            proper_names (list, optional): List of proper names extracted from the question.
            
        Returns:
            pandas.DataFrame: Matching rows, preferably a single row.
        """
        if self.df is None or self.df.empty:
            return None
        
        # Start with the full dataset
        filtered_df = self.df.copy()
        
        # If we have proper names, try to match them exactly first
        if proper_names and len(proper_names) > 0:
            # Try exact matches on both name and company columns for each proper name
            for proper_name in proper_names:
                # Check for exact matches first
                name_exact_matches = filtered_df[filtered_df['name'] == proper_name]
                company_exact_matches = filtered_df[filtered_df['company'] == proper_name]
                
                # If we have exact matches, prioritize those
                if not name_exact_matches.empty:
                    return name_exact_matches
                elif not company_exact_matches.empty:
                    return company_exact_matches
                
                # If no exact matches, try contains
                name_mask = filtered_df['name'].str.lower().str.contains(proper_name.lower(), na=False)
                company_mask = filtered_df['company'].str.lower().str.contains(proper_name.lower(), na=False)
                
                proper_name_matches = filtered_df[name_mask | company_mask]
                
                # If we found matches with the proper name, use those
                if not proper_name_matches.empty:
                    filtered_df = proper_name_matches
                    
                    # If we have a single match, we're done
                    if len(filtered_df) == 1:
                        return filtered_df
        
        # If we still have multiple rows or no proper name matches, use the entity name
        if entity_name and len(filtered_df) > 1:
            entity_name = entity_name.lower()
            
            # Depending on the query type, we'll search different columns
            if query_type == 'name' or query_type == 'company':
                # For name and company queries, we can directly search those columns
                column_to_search = query_type
                mask = filtered_df[column_to_search].str.lower().str.contains(entity_name, na=False)
                entity_matches = filtered_df[mask]
                
                # Only update our filtered results if we found matches
                if not entity_matches.empty:
                    filtered_df = entity_matches
            else:
                # For other query types, search both name and company columns
                name_mask = filtered_df['name'].str.lower().str.contains(entity_name, na=False)
                company_mask = filtered_df['company'].str.lower().str.contains(entity_name, na=False)
                combined_mask = name_mask | company_mask
                
                entity_matches = filtered_df[combined_mask]
                # Only update our filtered results if we found matches
                if not entity_matches.empty:
                    filtered_df = entity_matches
        
        # If we have no matches after filtering, return None
        if filtered_df.empty:
            return None
            
        # Apply further filters if we still have multiple matches
        if len(filtered_df) > 1:
            # If query is about a specific year, filter by latest year
            if query_type != 'year':
                latest_year = filtered_df['year'].max()
                year_filtered = filtered_df[filtered_df['year'] == latest_year]
                
                # Only use this filter if it gives us results
                if not year_filtered.empty:
                    filtered_df = year_filtered
            
            # If still multiple, prioritize records with higher sentiment impact (absolute value)
            if len(filtered_df) > 1 and query_type != 'sentiment':
                # Calculate absolute sentiment deviation from neutral (0)
                filtered_df['sentiment_impact'] = filtered_df['sentiment'].abs()
                # Sort by impact and get the top result
                filtered_df = filtered_df.sort_values('sentiment_impact', ascending=False).head(1)
        
        return filtered_df
    
    def _format_query_response(self, filtered_df, query_type, entity_name=None):
        """
        Format filtered dataframe into a human-readable response based on the query type.
        
        Args:
            filtered_df (pandas.DataFrame): Filtered DataFrame with matching rows.
            query_type (str): Type of query (column name).
            entity_name (str, optional): Entity name that was searched for.
            
        Returns:
            str: Formatted response.
        """
        if filtered_df is None or filtered_df.empty:
            if entity_name:
                return f"No information found for {entity_name} regarding {query_type}."
            else:
                return f"No information found regarding {query_type}."
        
        # Number of rows we found
        row_count = len(filtered_df)
        
        # For a single result, provide detailed information
        if row_count == 1:
            row = filtered_df.iloc[0]
            
            # Format varies by query type
            if query_type == 'name':
                response = f"{row['name']} was the CEO of {row['company']} in {row['year']}. "
                response += f"The turnover was {row['turnover_type']} with a sentiment score of {row['sentiment']}. "
                response += f"There were {row['turnover_before']} previous turnovers."
                return response
                
            elif query_type == 'company':
                response = f"Information about {row['company']}: "
                response += f"CEO {row['name']} in {row['year']}. "
                response += f"The turnover was {row['turnover_type']} with a sentiment score of {row['sentiment']}. "
                response += f"There were {row['turnover_before']} previous turnovers."
                return response
                
            elif query_type == 'turnover_type':
                response = f"The turnover of {row['name']} from {row['company']} in {row['year']} "
                response += f"was {row['turnover_type']} with a sentiment score of {row['sentiment']}."
                return response
                
            elif query_type == 'sentiment':
                response = f"The sentiment score for {row['name']}'s departure from {row['company']} "
                response += f"in {row['year']} was {row['sentiment']}. "
                response += f"This was a {row['turnover_type']} turnover."
                return response
                
            elif query_type == 'year':
                response = f"In {row['year']}, {row['name']} departed from {row['company']}. "
                response += f"It was a {row['turnover_type']} turnover with a sentiment score of {row['sentiment']}."
                return response
                
            elif query_type == 'turnover_before':
                response = f"Before {row['name']}'s {row['turnover_type']} departure from {row['company']} "
                response += f"in {row['year']}, there were {row['turnover_before']} previous turnovers."
                return response
                
            else:
                # Generic response for any other query type
                return f"Found information: {row.to_string()}"
                
        # For multiple results, provide a summary
        else:
            if query_type == 'name':
                names = filtered_df['name'].unique()
                response = f"Found information about {len(names)} CEOs: {', '.join(names[:5])}"
                if len(names) > 5:
                    response += f" and {len(names) - 5} others"
                return response
                
            elif query_type == 'company':
                companies = filtered_df['company'].unique()
                response = f"Found information about {len(companies)} companies: {', '.join(companies[:5])}"
                if len(companies) > 5:
                    response += f" and {len(companies) - 5} others"
                return response
                
            elif query_type == 'turnover_type':
                # Group by turnover type and count
                turnover_counts = filtered_df['turnover_type'].value_counts()
                response = "Turnover types found:\n"
                for type_name, count in turnover_counts.items():
                    response += f"- {type_name}: {count} instances\n"
                return response
                
            elif query_type == 'sentiment':
                avg_sentiment = filtered_df['sentiment'].mean()
                max_sentiment = filtered_df['sentiment'].max()
                min_sentiment = filtered_df['sentiment'].min()
                response = f"Sentiment analysis of {row_count} instances:\n"
                response += f"- Average sentiment: {avg_sentiment:.2f}\n"
                response += f"- Highest sentiment: {max_sentiment:.2f}\n"
                response += f"- Lowest sentiment: {min_sentiment:.2f}"
                return response
                
            elif query_type == 'year':
                year_counts = filtered_df['year'].value_counts().sort_index()
                response = "CEO departures by year:\n"
                for year, count in year_counts.items():
                    response += f"- {year}: {count} departures\n"
                return response
                
            elif query_type == 'turnover_before':
                avg_turnover = filtered_df['turnover_before'].mean()
                max_turnover = filtered_df['turnover_before'].max()
                response = f"Previous turnover statistics for {row_count} instances:\n"
                response += f"- Average previous turnovers: {avg_turnover:.2f}\n"
                response += f"- Maximum previous turnovers: {max_turnover}"
                return response
                
            else:
                # Generic response for any other query type with multiple results
                if row_count <= 5:
                    return f"Found {row_count} matching records:\n{filtered_df.to_string()}"
                else:
                    return f"Found {row_count} matching records. Here are the first 5:\n{filtered_df.head(5).to_string()}"
    
    def answer_question(self, question):
        """
        Answer a natural language question by querying the CEO turnover CSV,
        using proper name matching to narrow down to a single row when possible.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            str: Answer to the question.
        """
        if self.df is None:
            return "CSV file failed to load."
        
        # Preprocess the question
        processed_question = self._preprocess_question(question)
        
        # Identify query type (which maps to a column in our CSV)
        query_type = self._identify_query_type(processed_question)
        
        # Extract entity names including all proper names
        entity_name, proper_names = self._extract_entity_name(question)
        
        # Debug information
        debug_info = f"""
        Question: {question}
        Processed: {processed_question}
        Query type (column): {query_type}
        Primary entity: {entity_name}
        All proper names: {proper_names}
        """
        print(debug_info)
        
        # Find rows that match the query type and entity, using proper names to narrow down
        matching_rows = self._find_rows_by_query(query_type, entity_name, proper_names)
        
        # Format the response based on the query type and results
        answer = self._format_query_response(matching_rows, query_type, entity_name)
        
        return answer


def main():
    """
    Main function to run the CSV question answering system.
    """
    qa_system = CSVQuestionAnswerer()
    
    print("\nWelcome to CEO Insights")
    print("Type 'exit' to quit.")
    
    while True:
        question = input("\nWhat would you like to know?: ")
        
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        
        answer = qa_system.answer_question(question)
        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()