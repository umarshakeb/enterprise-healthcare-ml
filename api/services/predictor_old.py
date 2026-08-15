import pandas as pd
from api.services.model_loader import load_claim_model, load_risk_model
from monitoring.logger import log_prediction

def predict_risk_result(data):
    model = load_risk_model()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]

    log_prediction(
        model_name = "risk_model",
        model_version = "v1",
        input_data = data,
        prediction = prediction
    )

    response = {
        "prediction" : prediction
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        if hasattr(probabilities,"tolist"):
            response["probabilities"] = probabilities.tolist()
        else:
            response["probabilities"] = probabilities.to_list()

    return response

def predict_claim_result(data):
    model = load_claim_model()
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]

    log_prediction(
        model_name = "claim_model",
        model_version = "v1",
        input_data = data,
        prediction = prediction
    )

    response = {
        "prediction" : prediction
    }
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        if hasattr(probabilities,"tolist"):
            response["probabilities"] = probabilities.tolist()
        else:
            response["probabilities"] = probabilities.to_list()

    return response