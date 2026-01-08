import os
import numpy as np
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, "..", ".."))

DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
FINAL_REVIEWS_PATH = os.path.join(DATA_DIR, "final_reviews.csv")

HF_MODEL_ID = "shoyassi/sentiment-assessment"


@st.cache_data
def load_reviews(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_sentiment_model_from_hf(model_id: str):
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_sentiment(text: str, tokenizer, model, device):
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze().detach().cpu().numpy()

    pred_id = int(np.argmax(probs))

    id2label = getattr(model.config, "id2label", None) or {}
    label = id2label.get(pred_id)
    if not label:
        label = "Negative" if pred_id == 0 else "Positive"

    prob_map = {0: float(probs[0]), 1: float(probs[1])} if len(probs) >= 2 else {}

    return {
        "label": str(label),
        "prob_0": prob_map.get(0, None),
        "prob_1": prob_map.get(1, None),
        "confidence": float(np.max(probs)),
        "raw_probs": probs.tolist(),
    }


st.title("Sentiment")

reviews_df = load_reviews(FINAL_REVIEWS_PATH)

st.subheader("Dataset sentiment distribution (rating-derived)")
if "sentiment" in reviews_df.columns:
    sent = reviews_df["sentiment"].value_counts().reset_index()
    sent.columns = ["sentiment", "count"]
    st.dataframe(sent, use_container_width=True)
else:
    st.info("sentiment column not found in final_reviews.csv")

st.divider()

st.subheader("Interactive prediction (Transformer model)")

user_text = st.text_area("Enter a review text (Portuguese is OK):", height=120)

colA, colB = st.columns(2)
with colA:
    if st.button("Load model"):
        st.session_state["sentiment_loaded"] = True

with colB:
    if st.button("Unload model"):
        st.session_state["sentiment_loaded"] = False

if st.session_state.get("sentiment_loaded", False):
    try:
        tokenizer, model, device = load_sentiment_model_from_hf(HF_MODEL_ID)
        st.write("Model loaded from:", HF_MODEL_ID)
        st.write("Device:", str(device))
    except Exception as e:
        st.error("Failed to load model from Hugging Face.")
        st.exception(e)
        st.stop()

    if st.button("Predict"):
        if user_text.strip() == "":
            st.warning("Please enter some text first.")
        else:
            out = predict_sentiment(user_text, tokenizer, model, device)
            st.write("Prediction:", out["label"])
            st.write("Confidence:", round(out["confidence"], 4))

            if out["prob_0"] is not None and out["prob_1"] is not None:
                probs_df = pd.DataFrame(
                    {
                        "class": ["Negative", "Positive"],
                        "probability": [out["prob_0"], out["prob_1"]],
                    }
                )
                st.bar_chart(probs_df.set_index("class")["probability"])
            else:
                probs_df = pd.DataFrame(
                    {"class_id": list(range(len(out["raw_probs"]))), "probability": out["raw_probs"]}
                )
                st.bar_chart(probs_df.set_index("class_id")["probability"])
else:
    st.info("Click 'Load model' to enable prediction.")