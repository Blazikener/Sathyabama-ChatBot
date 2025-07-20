import pandas as pd
import os
from llama_index.experimental.query_engine import PandasQueryEngine

def get_pandas_engine(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    instruction_str = """\
1. Convert the query to executable Python code using Pandas.
2. The final line of code should be a Python expression that can be called with `eval()`.
3. PRINT ONLY THE EXPRESSION.
4. Handle missing data appropriately."""
    
    return PandasQueryEngine(
        df=df,
        verbose=True,
        instruction_str=instruction_str
    )