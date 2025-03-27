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
            
            # Define mappings between query types and row headers
            # This is the key modification - instead of identifying columns,
            # we'll map query types to specific row headers
            self.query_to_row_mapping = {
                'sentiment': ['Sentiment', 'Outcome', 'Result'],
                'turnover_type': ['Type', 'Category', 'Turnover Type'],
                'name': ['Name', 'CEO', 'Executive', 'Leader'],
                'date': ['Date', 'When', 'Time', 'Period'],
                'reason': ['Reason', 'Cause', 'Why'],
                # Add more mappings as needed for your specific CSV structure
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
        # Simple heuristic - if the first column of the first row contains
        # a string that matches any of our known header types
        if len(self.df) > 0:
            first_row = self.df.iloc[0]
            for header_list in self.query_to_row_mapping.values():
                for header in header_list:
                    for col in self.df.columns:
                        if isinstance(first_row[col], str) and header.lower() in first_row[col].lower():
                            return True
        return False
    
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
        Extract keywords from the question.
        
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
        
        # Add potential row headers as keywords
        header_keywords = []
        for header_list in self.query_to_row_mapping.values():
            for header in header_list:
                header_lower = header.lower()
                header_keywords.append(header_lower)
                parts = re.sub(r'[^a-zA-Z0-9\s]', ' ', header_lower).split()
                header_keywords.extend(parts)
        
        # Create a set of keywords from headers and question words
        all_keywords = set(words + header_keywords)
        
        return list(all_keywords)
    
    def _identify_query_type(self, question):
        """
        Identify the type of query based on the question.
        
        Args:
            question (str): The preprocessed question.
            
        Returns:
            str: Type of query
        """
        # Check for each query type using keywords
        query_type_scores = {}
        for query_type, headers in self.query_to_row_mapping.items():
            score = 0
            # Check for direct header matches
            for header in headers:
                if header.lower() in question:
                    score += 2
            
            # Check for related question words
            if query_type == 'sentiment' and any(w in question for w in ['happen', 'result', 'outcome']):
                score += 1
            elif query_type == 'turnover_type' and any(w in question for w in ['voluntary', 'involuntary', 'fired', 'quit']):
                score += 1
            elif query_type == 'name' and any(w in question for w in ['who', 'person', 'individual']):
                score += 1
            elif query_type == 'date' and any(w in question for w in ['when', 'time', 'day', 'month', 'year']):
                score += 1
            elif query_type == 'reason' and any(w in question for w in ['why', 'because', 'due to']):
                score += 1
            
            if score > 0:
                query_type_scores[query_type] = score
        
        # Select the query type with the highest score
        if query_type_scores:
            return max(query_type_scores.items(), key=lambda x: x[1])[0]
        
        # Default query type
        return 'filter'
    
    def _extract_entity_name(self, question):
        """
        Extract a potential entity name (like a CEO or company name) from the question.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            str or None: Extracted entity name, if any.
        """
        # Look for quoted strings which often contain names
        match = re.search(r'"([^"]+)"', question)
        if match:
            return match.group(1)
        
        match = re.search(r"'([^']+)'", question)
        if match:
            return match.group(1)
        
        # Look for capitalized words which might be names
        words = re.findall(r'\b[A-Z][a-z]*\b', question)
        if words:
            # Join consecutive capitalized words as they might form a name
            names = []
            current_name = []
            
            for word in words:
                current_name.append(word)
                if len(current_name) > 0 and len(words) > words.index(word) + 1:
                    if not words[words.index(word) + 1].istitle():
                        names.append(' '.join(current_name))
                        current_name = []
            
            if current_name:
                names.append(' '.join(current_name))
            
            if names:
                return names[0]
        
        return None
    
    def _find_row_by_header_and_entity(self, query_type, entity_name=None):
        """
        Find a row by matching the header type and optionally an entity name.
        
        Args:
            query_type (str): Type of query.
            entity_name (str, optional): Entity name to search for.
            
        Returns:
            pandas.Series or None: Matching row if found.
        """
        if self.df is None or self.df.empty:
            return None
        
        # Get possible headers for this query type
        possible_headers = self.query_to_row_mapping.get(query_type, [])
        
        # If we have a header row, search for the column containing the header
        if self.has_header_row:
            first_row = self.df.iloc[0]
            for col in self.df.columns:
                cell_value = str(first_row[col]).lower()
                if any(header.lower() in cell_value for header in possible_headers):
                    # Found a header match - now look for the entity if provided
                    if entity_name:
                        for i, row in self.df.iterrows():
                            if i == 0:  # Skip the header row
                                continue
                            # Check if the entity name appears in the row
                            row_values = ' '.join(str(val).lower() for val in row.values)
                            if entity_name.lower() in row_values:
                                return row
                    else:
                        # If no entity, return the first data row
                        if len(self.df) > 1:
                            return self.df.iloc[1]  # Return the row after the header
        
        # If no header row or no match found, search all cells for header and entity
        for i, row in self.df.iterrows():
            row_values = ' '.join(str(val).lower() for val in row.values)
            
            # Check if any header appears in the row
            if any(header.lower() in row_values for header in possible_headers):
                # If entity provided, check if it also appears in the row
                if entity_name:
                    if entity_name.lower() in row_values:
                        return row
                else:
                    return row  # Return the first row with a header match
        
        return None
    
    def _format_row_response(self, row, query_type):
        """
        Format a row into a human-readable response based on the query type.
        
        Args:
            row (pandas.Series): DataFrame row.
            query_type (str): Type of query.
            
        Returns:
            str: Formatted response.
        """
        if row is None:
            return f"No information found for this {query_type} query."
        
        # Get possible headers for this query type
        possible_headers = self.query_to_row_mapping.get(query_type, [])
        
        # Try to find the value corresponding to the header
        header_col = None
        header_value = None
        
        for col in self.df.columns:
            cell_value = str(row[col])
            if any(header.lower() in cell_value.lower() for header in possible_headers):
                header_col = col
                header_value = cell_value
                break
        
        if header_col is not None:
            # Find the adjacent column that might contain the answer
            cols = list(self.df.columns)
            header_idx = cols.index(header_col)
            
            # Try to find a value column (usually to the right of the header)
            if header_idx + 1 < len(cols):
                answer_col = cols[header_idx + 1]
                answer_value = row[answer_col]
                return f"{header_value}: {answer_value}"
            else:
                # If no adjacent column, return the whole row
                return f"Found information: {row.to_string()}"
        else:
            # If no specific header column is found, return the whole row
            return f"Related information:\n{row.to_string()}"
    
    def answer_question(self, question):
        """
        Answer a natural language question by looking up row headers in the CSV.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            str: Answer to the question.
        """
        if self.df is None:
            return "CSV file failed to load."
        
        # Preprocess the question
        processed_question = self._preprocess_question(question)
        
        # Extract keywords
        keywords = self._extract_keywords(processed_question)
        
        # Identify query type
        query_type = self._identify_query_type(processed_question)
        
        # Extract entity name if present
        entity_name = self._extract_entity_name(question)
        
        # Debug information
        debug_info = f"""
        Question: {question}
        Processed: {processed_question}
        Query type: {query_type}
        Entity name: {entity_name}
        """
        print(debug_info)
        
        # Find the row that matches the query type and entity
        matching_row = self._find_row_by_header_and_entity(query_type, entity_name)
        
        # Format the response
        answer = self._format_row_response(matching_row, query_type)
        
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