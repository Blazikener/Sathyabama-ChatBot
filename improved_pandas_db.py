import pandas as pd
import os
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import Settings
import logging

logger = logging.getLogger(__name__)

def validate_csv_file(csv_path: str):
    """Validate CSV file exists and has content"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Check if file is not empty
    if os.path.getsize(csv_path) == 0:
        raise ValueError(f"CSV file is empty: {csv_path}")
    
    logger.info(f"✓ CSV file validated: {csv_path}")

def get_pandas_engine(csv_path: str):
    """Enhanced pandas query engine with better error handling"""
    
    validate_csv_file(csv_path)
    
    try:
        # Load CSV with error handling
        df = pd.read_csv(csv_path)
        
        if df.empty:
            raise ValueError(f"CSV file contains no data: {csv_path}")
        
        logger.info(f"✓ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"✓ Columns: {list(df.columns)}")
        
        # Enhanced instruction string for better query handling
        instruction_str = f"""\
You are working with a pandas DataFrame containing {len(df)} rows and {len(df.columns)} columns.

Column names: {list(df.columns)}

Instructions:
1. Convert the user query to executable Python code using pandas operations on the DataFrame 'df'
2. Handle case-insensitive searches using .str.contains() with case=False when searching text
3. Use .str.lower() for exact string matching
4. For date/time queries, ensure proper parsing if applicable
5. Return results in a user-friendly format
6. The final line should be a Python expression that can be evaluated
7. Handle missing data appropriately using .fillna() or .dropna() when needed
8. For price/cost queries, format currency appropriately
9. PRINT ONLY THE FINAL EXPRESSION, no explanations

Example patterns:
- For "show me Monday food": df[df['Day'].str.lower() == 'monday']
- For "bus routes to Chennai": df[df['Route'].str.contains('Chennai', case=False, na=False)]
- For "cheapest food": df.loc[df['Price'].str.replace('₹','').astype(int).idxmin()]
"""
        
        # Create query engine with the current LLM from Settings
        query_engine = PandasQueryEngine(
            df=df,
            verbose=True,
            instruction_str=instruction_str,
            llm=Settings.llm  # Use the globally set LLM (Groq)
        )
        
        logger.info(f"✓ Pandas query engine created for {csv_path}")
        return query_engine
        
    except Exception as e:
        logger.error(f"Failed to create pandas engine for {csv_path}: {str(e)}")
        raise

def test_pandas_engine(query_engine, test_query="Show me all data"):
    """Test pandas query engine functionality"""
    try:
        response = query_engine.query(test_query)
        logger.info(f"✓ Pandas engine test successful")
        return True
    except Exception as e:
        logger.error(f"✗ Pandas engine test failed: {e}")
        return False

