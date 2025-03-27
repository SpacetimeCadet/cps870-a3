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
            
            # Define the reduced set of query types we'll support
            self.query_types = ['name', 'sentiment', 'turnover_type']
            
            # Store min and max years for year-based filtering
            self.min_year = self.df['year'].min()
            self.max_year = self.df['year'].max()
            
            # Define mappings between query types and actual column names in the CSV
            self.query_to_column_mapping = {
                'name': 'name',
                'sentiment': 'sentiment',
                'turnover_type': 'turnover_type'
            }
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            self.df = None
    
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
    
    def _identify_query_type(self, question):
        """
        Identify the type of query based on the question, limited to our reduced set of query types.
        
        Args:
            question (str): The preprocessed question.
            
        Returns:
            str: Type of query (name, sentiment, or turnover_type)
        """
        # Mapping of keywords to our reduced query types
        keyword_to_query_map = {
            # name keywords
            'who': 'name',
            'ceo': 'name',
            'executive': 'name',
            'person': 'name',
            'name': 'name',
            
            # sentiment keywords
            'sentiment': 'sentiment',
            'feeling': 'sentiment',
            'reaction': 'sentiment',
            'positive': 'sentiment',
            'negative': 'sentiment',
            'market': 'sentiment',
            'score': 'sentiment',
            'impact': 'sentiment',
            
            # turnover_type keywords
            'type': 'turnover_type',
            'voluntary': 'turnover_type',
            'involuntary': 'turnover_type',
            'fired': 'turnover_type',
            'resigned': 'turnover_type',
            'quit': 'turnover_type',
            'forced': 'turnover_type',
            'departure': 'turnover_type',
            'left': 'turnover_type',
            'exit': 'turnover_type'
        }
        
        # Check for name-specific question patterns first
        name_patterns = [
            r'\bwho\b',  # "Who left Apple?"
            r'\bname\b',  # "What's the name of the CEO who..."
            r'\bwhich\s+ceo\b',  # "Which CEO left in 2020?"
            r'\bwhich\s+person\b',  # "Which person was fired from Google?"
        ]
        
        for pattern in name_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                return 'name'
        
        # Split question into words
        words = question.split()
        
        # Check for keyword matches
        for word in words:
            if word in keyword_to_query_map:
                return keyword_to_query_map[word]
        
        # Look for capitalized proper names as a strong indicator this is a name query
        proper_names = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
        if proper_names:
            # If we have proper names and no other strong indicators, assume name query
            return 'name'
                
        # If no specific query type is identified, default to name
        return 'name'
    
    def _extract_entity_name(self, question):
        """
        Extract potential CEO names from the question with enhanced focus on proper names
        that are likely to be CEOs rather than companies.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            tuple: (primary_entity, list_of_proper_names)
                - primary_entity: Primary entity name (str or None)
                - list_of_proper_names: List of all proper names found (list)
        """
        proper_names = []
        primary_entity = None
        
        # Enhanced pattern to find CEO name indicators
        ceo_indicators = [
            r'CEO\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'chief executive\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'executive\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was|is|as)\s+(?:the\s+)?CEO',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:left|departed|exited|resigned|quit)'
        ]
        
        # Look for CEO name indicators first (highest priority)
        for pattern in ceo_indicators:
            matches = re.findall(pattern, question)
            if matches:
                proper_names.extend(matches)
                if not primary_entity:
                    primary_entity = matches[0]
        
        # Look for quoted strings which often contain specific names (high priority)
        quoted_matches = re.findall(r'"([^"]+)"', question)
        quoted_matches.extend(re.findall(r"'([^']+)'", question))
        
        if quoted_matches:
            if not primary_entity:
                primary_entity = quoted_matches[0]
            proper_names.extend(quoted_matches)
        
        # Look for capitalized words that might be CEO names
        # This regex finds sequences of capitalized words
        capitalized_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', question)
        if capitalized_matches:
            proper_names.extend(capitalized_matches)
            # If we don't have a primary entity yet, use the match with fewest words
            # (CEO names are often shorter than company names)
            if not primary_entity and capitalized_matches:
                primary_entity = min(capitalized_matches, key=lambda x: len(x.split()))
        
        # If we have a list of proper names but no primary entity, use the first one
        if not primary_entity and proper_names:
            primary_entity = proper_names[0]
            
        # If still no entity, look for any capitalized word
        if not primary_entity:
            cap_words = re.findall(r'\b([A-Z][a-z]+)\b', question)
            if cap_words:
                primary_entity = cap_words[0]
                proper_names.append(cap_words[0])
                
        # If absolutely no entity found, use keywords that might indicate a person
        if not primary_entity:
            person_keywords = ['who', 'ceo', 'executive', 'person', 'leader', 'chief']
            for word in question.lower().split():
                if word in person_keywords:
                    primary_entity = word
                    break
        
        # Remove duplicates while preserving order
        unique_proper_names = []
        for name in proper_names:
            if name not in unique_proper_names:
                unique_proper_names.append(name)
        
        return primary_entity, unique_proper_names
    
    def _extract_years(self, question):
        """
        Extract potential year references from the question.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            list: List of years found in the question.
        """
        # Extract 4-digit years (2000-2025 range)
        year_pattern = r'\b(20\d{2})\b'
        years_found = re.findall(year_pattern, question)
        
        # Convert to integers
        years = [int(year) for year in years_found]
        
        # Look for decade references
        decade_patterns = {
            r'\b(2000s)\b': range(2000, 2010),
            r'\b(2010s)\b': range(2010, 2020),
            r'\b(2020s)\b': range(2020, 2025)  # Up to our current max
        }
        
        for pattern, year_range in decade_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                years.extend(list(year_range))
        
        # Look for relative time references
        relative_patterns = {
            r'\blast\s+year\b': [self.max_year - 1],
            r'\bprevious\s+year\b': [self.max_year - 1],
            r'\bcurrent\s+year\b': [self.max_year],
            r'\bthis\s+year\b': [self.max_year],
            r'\brecent\b': list(range(self.max_year - 2, self.max_year + 1)),
            r'\boldest\b': [self.min_year],
            r'\bearliest\b': [self.min_year]
        }
        
        for pattern, year_values in relative_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                years.extend(year_values)
        
        # Look for before/after year patterns
        before_after_patterns = [
            (r'\bbefore\s+(20\d{2})\b', lambda y: list(range(self.min_year, int(y)))),
            (r'\bprior\s+to\s+(20\d{2})\b', lambda y: list(range(self.min_year, int(y)))),
            (r'\bafter\s+(20\d{2})\b', lambda y: list(range(int(y) + 1, self.max_year + 1))),
            (r'\bsince\s+(20\d{2})\b', lambda y: list(range(int(y), self.max_year + 1)))
        ]
        
        for pattern, year_func in before_after_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                years.extend(year_func(match))
        
        # Remove duplicates
        return sorted(list(set(years)))
    
    def _find_rows_by_query(self, query_type, entity_name=None, proper_names=None, years=None):
        """
        Find rows specifically matching the CEO name in the query.
        
        Args:
            query_type (str): Column to query (name, sentiment, or turnover_type).
            entity_name (str, optional): Primary entity name to search for.
            proper_names (list, optional): List of proper names extracted from the question.
            years (list, optional): List of years extracted from the question.
            
        Returns:
            pandas.DataFrame: Only the row(s) matching the specific CEO name in the query.
        """
        if self.df is None or self.df.empty:
            return None
        
        # Start with the full dataset
        filtered_df = self.df.copy()
        
        # First priority: Find exact CEO name matches
        ceo_matches = None
        
        # Use proper names as potential CEO names, prioritizing them
        if proper_names and len(proper_names) > 0:
            for proper_name in proper_names:
                # Try exact match first
                exact_match = filtered_df[filtered_df['name'] == proper_name]
                if not exact_match.empty:
                    ceo_matches = exact_match
                    break
                
                # Then try partial match on the name column only
                partial_match = filtered_df[filtered_df['name'].str.lower().str.contains(proper_name.lower(), na=False)]
                if not partial_match.empty:
                    ceo_matches = partial_match
                    break
        
        # If no matches with proper names, try the entity name
        if ceo_matches is None and entity_name:
            # Try exact match first
            exact_match = filtered_df[filtered_df['name'] == entity_name]
            if not exact_match.empty:
                ceo_matches = exact_match
            else:
                # Then try partial match on the name column only
                partial_match = filtered_df[filtered_df['name'].str.lower().str.contains(entity_name.lower(), na=False)]
                if not partial_match.empty:
                    ceo_matches = partial_match
        
        # If we found CEO matches, apply year filtering if specified
        if ceo_matches is not None and not ceo_matches.empty:
            if years and len(years) > 0:
                year_matches = ceo_matches[ceo_matches['year'].isin(years)]
                
                # Only use year filter if it gives results
                if not year_matches.empty:
                    ceo_matches = year_matches
                else:
                    # If no direct year matches, find the closest year
                    ceo_matches['year_distance'] = ceo_matches['year'].apply(
                        lambda x: min(abs(x - y) for y in years)
                    )
                    # Sort by closest year and take the top match
                    ceo_matches = ceo_matches.sort_values('year_distance').head(1)
                    # Remove the temporary column
                    ceo_matches = ceo_matches.drop('year_distance', axis=1)
            
            # If multiple matches for the same CEO, prioritize the most recent
            if len(ceo_matches) > 1:
                ceo_matches = ceo_matches.sort_values('year', ascending=False).head(1)
            
            return ceo_matches
        
        # If no CEO matches, we return None
        return None
    
    def _format_query_response(self, filtered_df, query_type, entity_name=None, years=None):
        """
        Format filtered dataframe into a human-readable response based on the query type.
        Focuses on providing detailed information about the specific CEO found.
        
        Args:
            filtered_df (pandas.DataFrame): Filtered DataFrame with matching rows.
            query_type (str): Type of query (name, sentiment, or turnover_type).
            entity_name (str, optional): Entity name that was searched for.
            years (list, optional): Years that were extracted from the query.
            
        Returns:
            str: Formatted response about the specific CEO.
        """
        if filtered_df is None or filtered_df.empty:
            year_str = ""
            if years and len(years) > 0:
                year_str = f" in {', '.join(map(str, years))}"
                
            if entity_name:
                return f"No CEO information found for {entity_name}{year_str}."
            else:
                return f"No specific CEO information found{year_str}."
        
        # We should only have one CEO at this point due to our filtering, but handle multiple just in case
        if len(filtered_df) > 1:
            # Get the most recent record
            row = filtered_df.sort_values('year', ascending=False).iloc[0]
        else:
            row = filtered_df.iloc[0]
        
        # Build a comprehensive response about this CEO, regardless of query type
        response = f"{row['name']} was the CEO of {row['company']} "
        
        # Add year information
        if years and len(years) > 0:
            # If the year was specifically queried
            response += f"and departed in {row['year']}. "
        else:
            response += f"until {row['year']}. "
        
        # Add turnover information
        response += f"The departure was {row['turnover_type']} with a "
        
        # Add sentiment information with interpretation
        sentiment_val = row['sentiment']
        if sentiment_val > 0.5:
            sentiment_desc = "very positive"
        elif sentiment_val > 0:
            sentiment_desc = "positive"
        elif sentiment_val > -0.5:
            sentiment_desc = "slightly negative"
        else:
            sentiment_desc = "strongly negative"
            
        response += f"{sentiment_desc} market reaction (sentiment score: {sentiment_val}). "
        
        # Add previous turnover information
        if row['turnover_before'] == 0:
            response += f"There were no previous CEO turnovers at {row['company']} before this."
        elif row['turnover_before'] == 1:
            response += f"There was 1 previous CEO turnover at {row['company']} before this."
        else:
            response += f"There were {row['turnover_before']} previous CEO turnovers at {row['company']} before this."
        
        return response
    
    def answer_question(self, question):
        """
        Answer a natural language question by finding and returning information
        about the specific CEO mentioned in the query.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            str: Answer focused on the specific CEO mentioned.
        """
        if self.df is None:
            return "CSV file failed to load."
        
        # Preprocess the question
        processed_question = self._preprocess_question(question)
        
        # Identify query type (which maps to a column in our CSV)
        query_type = self._identify_query_type(processed_question)
        
        # Extract entity names with enhanced focus on CEO names
        entity_name, proper_names = self._extract_entity_name(question)
        
        # Extract years from the question
        years = self._extract_years(question)
        
        # Debug information
        debug_info = f"""
        Question: {question}
        Processed: {processed_question}
        Query type (column): {query_type}
        Primary entity (likely CEO): {entity_name}
        Potential CEO names: {proper_names}
        Years mentioned: {years}
        """
        print(debug_info)
        
        # Find the CEO-specific record that matches the query
        ceo_record = self._find_rows_by_query(query_type, entity_name, proper_names, years)
        
        # Add the proper_names parameter when formatting the response
        answer = self._format_query_response(ceo_record, query_type, entity_name, years)
        
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