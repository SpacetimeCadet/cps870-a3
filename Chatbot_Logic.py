import pandas as pd
import re
import string
from collections import Counter
import numpy as np

# Global Variables
input_file = "input.csv"


class CSVQuestionAnswerer:
    """
    A simple question answering system that uses pandas to query a CSV file
    based on natural language questions.
    """
    
    def __init__(self, csv_path=input_file):
        """
        Initialize the question answering system with a CSV file.
        
        Args:
            csv_path (str): Path to the CSV file to be queried.
        """
        self.csv_path = csv_path
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Successfully loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns.")
            print(f"Columns: {', '.join(self.df.columns)}")
            
            # Store column types for better query handling
            self.column_types = {col: str(self.df[col].dtype) for col in self.df.columns}
            
            # Store some sample data for understanding the content
            self.sample_data = self.df.head(3)
            
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
                      'to', 'of', 'for', 'with', 'what', 'which', 'who', 'where',
                      'when', 'how', 'many', 'much', 'do', 'does', 'list', 'tell',
                      'me', 'show', 'give', 'find', 'all', 'that', 'have', 'has']
        
        words = [word for word in question.split() if word not in stop_words]
        
        # Add column names as potential keywords
        column_keywords = []
        for col in self.df.columns:
            col_lower = col.lower()
            # Include both the full column name and its individual parts
            column_keywords.append(col_lower)
            parts = re.sub(r'[^a-zA-Z0-9\s]', ' ', col_lower).split()
            column_keywords.extend(parts)
        
        # Create a set of keywords from column names, question words, and their combinations
        all_keywords = set(words + column_keywords)
        
        return list(all_keywords)
    
    def _identify_query_type(self, question):
        """
        Identify the type of query based on the question.
        
        Args:
            question (str): The preprocessed question.
            
        Returns:
            str: Type of query (e.g., 'sum', 'average', 'count', 'max', 'min', 'filter').
        """
        # Aggregate functions
        if any(w in question for w in ['sum', 'total', 'add']):
            return 'sum'
        elif any(w in question for w in ['average', 'avg', 'mean']):
            return 'average'
        elif any(w in question for w in ['count', 'how many', 'number of']):
            return 'count'
        elif any(w in question for w in ['maximum', 'max', 'highest', 'largest', 'most']):
            return 'max'
        elif any(w in question for w in ['minimum', 'min', 'lowest', 'smallest', 'least']):
            return 'min'
        elif any(w in question for w in ['greater than', 'more than', 'bigger than', 'larger than']):
            return 'greater_than'
        elif any(w in question for w in ['less than', 'smaller than', 'lower than']):
            return 'less_than'
        elif any(w in question for w in ['top', 'bottom']):
            return 'top_n'
        
        # Default query type is filter
        return 'filter'
    
    def _identify_columns(self, question, keywords):
        """
        Identify relevant columns based on question and keywords.
        
        Args:
            question (str): The preprocessed question.
            keywords (list): Extracted keywords.
            
        Returns:
            list: Relevant column names.
        """
        relevant_columns = []
        
        # Score each column based on how many of its words appear in the question
        column_scores = {}
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Direct match
            if col_lower in question:
                relevant_columns.append(col)
                continue
            
            # Word-by-word match
            col_words = re.sub(r'[^a-zA-Z0-9\s]', ' ', col_lower).split()
            match_score = sum(1 for word in col_words if word in keywords)
            
            # Only consider columns with at least one matching word
            if match_score > 0:
                column_scores[col] = match_score
        
        # Add columns with the highest scores
        if column_scores:
            max_score = max(column_scores.values())
            for col, score in column_scores.items():
                if score == max_score and col not in relevant_columns:
                    relevant_columns.append(col)
        
        # If no columns are found, try to infer from question and data types
        if not relevant_columns:
            if 'average' in question or 'sum' in question or any(w in question for w in ['most', 'least', 'highest', 'lowest']):
                # For aggregate functions, prefer numeric columns
                for col, dtype in self.column_types.items():
                    if 'int' in dtype or 'float' in dtype:
                        relevant_columns.append(col)
            elif any(w in question for w in ['date', 'when', 'time', 'year', 'month', 'day']):
                # For time-related questions, prefer date columns
                for col in self.df.columns:
                    if any(w in col.lower() for w in ['date', 'time', 'year', 'month', 'day']):
                        relevant_columns.append(col)
        
        # If still no columns, include all columns
        if not relevant_columns:
            relevant_columns = list(self.df.columns)
        
        return relevant_columns
    
    def _extract_numeric_value(self, question):
        """
        Extract numeric values from the question.
        
        Args:
            question (str): The natural language question.
            
        Returns:
            float or None: Extracted numeric value, if any.
        """
        # Look for numeric patterns in the question
        numbers = re.findall(r'\d+\.?\d*', question)
        if numbers:
            return float(numbers[0])
        return None
    
    def _extract_string_value(self, question, keywords):
        """
        Extract string values from the question.
        
        Args:
            question (str): The natural language question.
            keywords (list): Extracted keywords.
            
        Returns:
            str or None: Extracted string value, if any.
        """
        # Look for patterns like "contains X" or "equals X"
        patterns = [
            r'contains\s+([a-zA-Z0-9_\s]+)',
            r'equals\s+([a-zA-Z0-9_\s]+)',
            r'equal to\s+([a-zA-Z0-9_\s]+)',
            r'is\s+([a-zA-Z0-9_\s]+)',
            r'where\s+([a-zA-Z0-9_\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                return match.group(1).strip()
        
        # Look for quoted strings
        match = re.search(r'"([^"]+)"', question)
        if match:
            return match.group(1)
        
        match = re.search(r"'([^']+)'", question)
        if match:
            return match.group(1)
        
        # Find words that are not column names or common keywords
        column_words = []
        for col in self.df.columns:
            column_words.extend(col.lower().split())
        
        common_words = ['what', 'which', 'who', 'where', 'when', 'how', 'many', 
                        'show', 'display', 'find', 'get', 'list', 'tell']
        
        question_words = question.split()
        potential_values = [word for word in question_words 
                            if word not in column_words 
                            and word not in common_words
                            and word not in keywords]
        
        if potential_values:
            return potential_values[0]
        
        return None
    
    def _execute_query(self, query_type, columns, numeric_value=None, string_value=None):
        """
        Execute the query based on identified parameters.
        
        Args:
            query_type (str): Type of query.
            columns (list): Columns to query.
            numeric_value (float, optional): Numeric value for comparison.
            string_value (str, optional): String value for comparison.
            
        Returns:
            str: Query result.
        """
        if not columns:
            return "No relevant columns identified."
        
        try:
            # For aggregate queries
            if query_type == 'sum':
                numeric_cols = [col for col in columns if 'int' in self.column_types[col] or 'float' in self.column_types[col]]
                if not numeric_cols:
                    return "No numeric columns found for summing."
                results = {col: self.df[col].sum() for col in numeric_cols}
                return f"Sum: {results}"
                
            elif query_type == 'average':
                numeric_cols = [col for col in columns if 'int' in self.column_types[col] or 'float' in self.column_types[col]]
                if not numeric_cols:
                    return "No numeric columns found for averaging."
                results = {col: self.df[col].mean() for col in numeric_cols}
                return f"Average: {results}"
                
            elif query_type == 'count':
                if string_value:
                    counts = {}
                    for col in columns:
                        if self.df[col].dtype == 'object':  # String column
                            count = self.df[self.df[col].str.contains(string_value, case=False, na=False)].shape[0]
                            counts[col] = count
                    if counts:
                        return f"Count of records containing '{string_value}': {counts}"
                
                return f"Total records: {len(self.df)}"
                
            elif query_type == 'max':
                numeric_cols = [col for col in columns if 'int' in self.column_types[col] or 'float' in self.column_types[col]]
                if not numeric_cols:
                    return "No numeric columns found for maximum."
                results = {col: self.df[col].max() for col in numeric_cols}
                return f"Maximum: {results}"
                
            elif query_type == 'min':
                numeric_cols = [col for col in columns if 'int' in self.column_types[col] or 'float' in self.column_types[col]]
                if not numeric_cols:
                    return "No numeric columns found for minimum."
                results = {col: self.df[col].min() for col in numeric_cols}
                return f"Minimum: {results}"
                
            elif query_type == 'greater_than':
                if numeric_value is None:
                    return "No value specified for comparison."
                numeric_cols = [col for col in columns if 'int' in self.column_types[col] or 'float' in self.column_types[col]]
                if not numeric_cols:
                    return "No numeric columns found for comparison."
                filtered_data = self.df[self.df[numeric_cols[0]] > numeric_value]
                return f"Records where {numeric_cols[0]} > {numeric_value}:\n{filtered_data.head(5).to_string()}\n... and {len(filtered_data) - 5} more records" if len(filtered_data) > 5 else filtered_data.to_string()
                
            elif query_type == 'less_than':
                if numeric_value is None:
                    return "No value specified for comparison."
                numeric_cols = [col for col in columns if 'int' in self.column_types[col] or 'float' in self.column_types[col]]
                if not numeric_cols:
                    return "No numeric columns found for comparison."
                filtered_data = self.df[self.df[numeric_cols[0]] < numeric_value]
                return f"Records where {numeric_cols[0]} < {numeric_value}:\n{filtered_data.head(5).to_string()}\n... and {len(filtered_data) - 5} more records" if len(filtered_data) > 5 else filtered_data.to_string()
                
            elif query_type == 'top_n':
                n = int(numeric_value) if numeric_value else 5
                numeric_cols = [col for col in columns if 'int' in self.column_types[col] or 'float' in self.column_types[col]]
                if not numeric_cols:
                    return "No numeric columns found for top/bottom ranking."
                if 'top' in query_type:
                    sorted_data = self.df.sort_values(by=numeric_cols[0], ascending=False).head(n)
                    return f"Top {n} records by {numeric_cols[0]}:\n{sorted_data.to_string()}"
                else:
                    sorted_data = self.df.sort_values(by=numeric_cols[0], ascending=True).head(n)
                    return f"Bottom {n} records by {numeric_cols[0]}:\n{sorted_data.to_string()}"
            
            # Default: filter or show relevant columns
            else:
                if string_value:
                    # Try to find matches in string columns
                    str_cols = [col for col in columns if self.df[col].dtype == 'object']
                    if str_cols:
                        # Create a filter condition that matches the string value in any of the string columns
                        filter_condition = False
                        for col in str_cols:
                            filter_condition = filter_condition | self.df[col].str.contains(string_value, case=False, na=False)
                        
                        filtered_data = self.df[filter_condition]
                        if not filtered_data.empty:
                            return f"Filtered results containing '{string_value}':\n{filtered_data.head(5).to_string()}\n... and {len(filtered_data) - 5} more records" if len(filtered_data) > 5 else filtered_data.to_string()
                    
                # If no string value or no matches, show a sample of the relevant columns
                return f"Sample of relevant columns:\n{self.df[columns].head(5).to_string()}"
                
        except Exception as e:
            return f"Error executing query: {e}"
    
    def answer_question(self, question):
        """
        Answer a natural language question by querying the CSV file.
        
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
        
        # Identify relevant columns
        columns = self._identify_columns(processed_question, keywords)
        
        # Extract values for filtering
        numeric_value = self._extract_numeric_value(processed_question)
        string_value = self._extract_string_value(processed_question, keywords)
        
        # Debug information
        print(f"Question: {question}")
        print(f"Processed: {processed_question}")
        print(f"Keywords: {keywords}")
        print(f"Query type: {query_type}")
        print(f"Columns: {columns}")
        print(f"Numeric value: {numeric_value}")
        print(f"String value: {string_value}")
        
        # Execute the query
        answer = self._execute_query(query_type, columns, numeric_value, string_value)
        
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