import os
import multiprocessing as mp

import pandas as pd
from pyprojroot import here


lock = mp.Lock()

def log_token_usage(result, file_name=here('struct_vs_unstruct/logs/non_self_synthesis/evals/bbh/token_usage_log.csv')):
    # Check if the file already exists
    if os.path.exists(file_name):
        with lock:
            # Load the existing file
            df = pd.read_csv(file_name)
    else:
        # Create a new DataFrame with the appropriate columns
        df = pd.DataFrame(columns=["prompt_tokens", "completion_tokens", "total_tokens"])

    token_data = result.response_metadata.get("token_usage", {})

    # Create a DataFrame for the new token usage data
    new_row_df = pd.DataFrame([{
        "prompt_tokens": token_data.get('prompt_tokens', 0),
        "completion_tokens": token_data.get('completion_tokens', 0),
        "total_tokens": token_data.get('total_tokens', 0)
    }])

    # Use pd.concat to append the new row to the existing DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)

    with lock:
        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_name, index=False)