import os 
import json
import hashlib
import joblib
import pandas as pd
from datetime import datetime

# Utility functions for health and claim project

# Get base directory
def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__),".."))

# Fetch the constructed .csv data file
def get_model_table_path():
    return os.path.join(get_base_dir(), "outputs", "model_table_enriched.csv")

# Function to load the table
def load_model_table():
    model_table_path  = get_model_table_path()
    if not os.path.exists(model_table_path):
        raise FileNotFoundError(f"model_table_enriched.csv not found at : {model_table_path}")

    return pd.read_csv(model_table_path)

# Function to split data based on time-split. Splitting is performed on the specified sort column
def time_based_split(df,sort_column,split_ratio=0.8):
    if sort_column not in df.columns:
        raise ValueError(f"Sort column : {sort_column} not found in dataframe")
    
    df = df.copy()
    df[sort_column] = pd.to_datetime(df[sort_column])
    df = df.sort_values(sort_column).reset_index(drop=True)
    split_idx = int(len(df)*split_ratio)
    train_df = df[:split_idx].copy()
    test_df = df[split_idx:].copy()
    
    return train_df,test_df

# Save feature schema
def save_feature_schema_full(config):
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir,"outputs")
    os.makedirs(output_dir, exist_ok=True)
    schema_path  = os.path.join(output_dir,"feature_schema.json")

    schema = {
        "risk_model_features"  : config["risk"]["input_features"],
        "claim_model_features" : config["claim"]["input_features"],
        "risk_target"          : config["risk"]["target_column"],
        "claim_target"         : config["claim"]["target_column"],
        "risk_time_column"     : config["risk"]["sort_column"],
        "claim_time_column"    : config["claim"]["sort_column"],
        "split_strategy"       : "earliest 80 percent train, latest 20 percet test"
    }

    with open(schema_path, "w", encoding="utf-8") as f:
        json.dumps(schema,f, indent=4)
    
    return schema_path


# Load the schema file from output's directory
def load_feature_schema():
    base_dir = get_base_dir()
    schema_path = os.path.join(base_dir,"outputs","feature_schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"feature_schema.json not found at {schema_path}")
    
    with open(schema_path,"r",encoding="utf-8") as f:
        return json.load(f)


# Function to save trained model to local file in models directory using joblib
def save_local_model(model, file_name):
    base_dir  = get_base_dir()
    models_dir = os.path.join(base_dir,"models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, file_name)
    joblib.dump(model,model_path)

    return model_path

# Function to load trained model from models directory
def load_local_model(file_name):
    base_dir = get_base_dir()
    model_path = os.path.join(base_dir,"models",file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at : {model_path}")
    return joblib.load(model_path)

# This function generates a SHA-256 hash of the input payload dictionary.
# It first converts the dictionary to a JSON string with sorted keys to ensure consistent hashing,
# and then computes the hash of the string.
def hash_input(payload: dict) -> str:
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode()).hexdigest()


# This function writes a prediction log entry to a log file in the logs directory.
# Each log entry includes the timestamp, model name, model version, input hash, and the prediction result.
# The log is stored in JSON format, with one entry per line.
def write_prediction_log(model_name, model_version, input_hash,prediction):
    base_dir = get_base_dir()
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "predictions.log")

    prediction_log = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "input_hash": input_hash,
        "prediction": prediction
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction_log) + "\n")

    return log_file