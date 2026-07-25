from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Function to pick the raw input features from config.py
def get_input_feature_columns(config):
    return config['input_features'] 


# Function to build preprocessing pipeline using numeric and categorical features from config.py
def get_preprocessor(config):
    numeric_features = config['numeric_features']
    categorical_features = config['categorical_features']

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

# Function to build complete sklearn pipeline
def build_model_pipeline(config):
    preprocessor = get_preprocessor()
    model = RandomForestClassifier(
        n_estimators=config['params']['n_estimators'],
        max_depth = config['params']['max_depth'],
        min_samples_split = config['params']['min_samples_split'],
        min_samples_leaf = config['params']['min_samples_leaf'],
        class_weight = config['params']['class_weight'],
        random_state= 42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return pipeline

