from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tabpfn_client import TabPFNClassifier, init
import pandas as pd
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

import tabpfn_client

# Use environment variables
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


# --- TabPFN Setup ---
TABPFN_API_TOKEN = os.getenv("TABPFN_API_TOKEN")

tabpfn_client.set_access_token(TABPFN_API_TOKEN)

init()
clf = TabPFNClassifier(api_token=TABPFN_API_TOKEN)



# --- Load and preprocess dataset ---
data = pd.read_csv("predictions.csv")


for col in ["fever", "vomiting", "cough", "sore_throat", "fatigue"]:
    data[col] = data[col].map({"yes": 1, "no": 0})


target_map = {"video": 0, "in_person": 1, "self_care": 2}
data["consult_type_num"] = data["consult_type"].map(target_map)

# Features and labels
feature_cols = ["age", "fever", "vomiting", "cough", "sore_throat", "fatigue"]
X_train = data[feature_cols]
y_train = data["consult_type_num"].values

# Train the classifier
clf.fit(X_train, y_train)

# Reverse mapping for prediction decoding
inv_target_map = {v: k for k, v in target_map.items()}

# --- FastAPI Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SymptomInput(BaseModel):
    age: int
    fever: bool
    vomiting: bool
    cough: bool
    sore_throat: bool
    fatigue: bool

@app.post("/predict")
def predict(symptoms: SymptomInput):
    try:
        # Prepare input data for model
        data = [[
            symptoms.age,
            int(symptoms.fever),
            int(symptoms.vomiting),
            int(symptoms.cough),
            int(symptoms.sore_throat),
            int(symptoms.fatigue),
        ]]
        df = pd.DataFrame(data, columns=feature_cols)

        # Make prediction
        prediction = clf.predict(df)[0]
        proba_all = clf.predict_proba(df)[0]
        proba = proba_all[prediction]        
        predicted_consult = inv_target_map[prediction]

        # Construct prompt for GPT
        user_symptoms = ", ".join([
            k.replace("_", " ")
            for k, v in symptoms.dict().items()
            if (v and k != "age")
        ])

        prompt = (
            f"The user is {symptoms.age} years old and has: {user_symptoms}. "
            f"The model predicts the consultation type: {predicted_consult}.\n"
            f"Give medical advice in simple language."
        )

        # GPT-4 Response
        response = client.chat.completions.create(
             model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
             messages=[{"role": "user", "content": prompt}],
             max_tokens=150
            )
        gpt_text = response.choices[0].message.content.strip()

        return {
            "consult_type": predicted_consult,
            "probability": round(float(proba), 4),
            "probabilities": {
                "video": round(float(proba_all[0]), 4),
                "in_person": round(float(proba_all[1]), 4),
                "self_care": round(float(proba_all[2]), 4),
            },
            "gpt_advice": gpt_text
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
