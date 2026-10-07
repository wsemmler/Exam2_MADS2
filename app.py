# import streamlit as st
# import pandas as pd
# import joblib

# # Page configuration

# st.set_page_config(
#     page_title="Cyber Threat Severity Prediction",
#     page_icon="🛡️",
#     layout="wide"
# )

# # White background + simple styling

# st.markdown("""
# <style>
#     .stApp {
#         background-color: white;
#     }

#     h1, h2, h3, p, label {
#         color: #222222 !important;
#     }

#     .stTabs [data-baseweb="tab-list"] {
#         gap: 10px;
#     }

#     .stTabs [data-baseweb="tab"] {
#         color: #333333;
#         background-color: #f5f5f5;
#         border-radius: 8px 8px 0px 0px;
#         padding: 10px 20px;
#     }

#     .stTabs [aria-selected="true"] {
#         background-color: #e8e8e8;
#         color: #000000 !important;
#     }
# </style>
# """, unsafe_allow_html=True)


# # Load model

# model = joblib.load("model/xgb_model.joblib")
# label_encoder = joblib.load("model/label_encoder.joblib")


# # Title

# st.title("Cyber Threat Severity Prediction")
# st.write(
#     "Use the trained XGBoost model to estimate the severity "
#     "of cybersecurity events.")


# # TWO TABS

# tab1, tab2 = st.tabs([
#     "Existing Event",
#     "New Event"
# ])


# # TAB 1 — EXISTING EVENT

# with tab1:

#     st.header("Predict Severity from Existing Event")

#     st.write(
#         "Enter an existing Event ID from the dataset."
#     )

#     event_id = st.text_input(
#         "Event ID",
#         value="EVT_004660"
#     )

#     if st.button("Predict Existing Event"):

#         df = pd.read_csv(
#             "data/Federated_Cyber_Threat_Intelligence_Dataset.csv"
#         )

#         feature_columns = X.columns.tolist()

#         row = df[df["event_id"] == event_id]

#         if row.empty:

#             st.error("Event ID not found.")

#         else:

#             X_single = row[feature_columns]

#             probabilities = model.predict_proba(X_single)[0]

#             results = dict(
#                 zip(
#                     label_encoder.classes_,
#                     probabilities
#                 )
#             )

#             prediction = label_encoder.classes_[
#                 probabilities.argmax()
#             ]

#             st.success(
#                 f"Predicted Severity: {prediction}"
#             )

#             st.subheader("Severity Probabilities")

#             for severity, probability in results.items():

#                 st.write(
#                     f"**{severity}**: {probability:.2%}"
#                 )

#                 st.progress(float(probability))


# # TAB 2 — NEW EVENT

# with tab2:

#     st.header("Predict Severity from New Values")

#     st.write(
#         "Enter selected threat characteristics to estimate "
#         "the severity of a new event."
#     )

#     network_zone = st.number_input(
#         "Network Zone",
#         value=2
#     )

#     behavioral_score = st.number_input(
#         "Behavioral Risk Score",
#         value=78.5
#     )

#     attack_freq = st.number_input(
#         "Historical Attack Frequency",
#         value=12
#     )

#     failed_logins = st.number_input(
#         "Failed Login Count",
#         value=4
#     )

#     if st.button("Predict New Event"):

#         features = model.feature_names_in_

#         X_single = pd.DataFrame(
#             [[None] * len(features)],
#             columns=features
#         )

#         X_single.loc[
#             0,
#             [
#                 "network_zone",
#                 "behavioral_risk_score",
#                 "historical_attack_frequency",
#                 "failed_login_count"
#             ]
#         ] = [
#             network_zone,
#             behavioral_score,
#             attack_freq,
#             failed_logins
#         ]

#         X_single = X_single.fillna(0)

#         probabilities = model.predict_proba(X_single)[0]

#         results = dict(
#             zip(
#                 label_encoder.classes_,
#                 probabilities
#             )
#         )

#         prediction = label_encoder.classes_[
#             probabilities.argmax()
#         ]

#         st.success(
#             f"Predicted Severity: {prediction}"
#         )

#         st.subheader("Severity Probabilities")

#         for severity, probability in results.items():

#             st.write(
#                 f"**{severity}**: {probability:.2%}"
#             )

#             st.progress(float(probability))

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("model/xgb_grid_model.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

df = pd.read_csv("data/Federated_Cyber_Threat_Intelligence_Dataset.csv")

feature_columns = model.feature_names_in_.tolist()

# 1st function (existed event)
def predict_threat_probability(event_id):
    row = df[df["event_id"] == event_id]
    if row.empty:
        return None
    X_single = row[feature_columns]
    probabilities = model.predict_proba(X_single)[0]
    return probabilities

# 2nd function (new event)
def predict_raw_threat_probability(byte_rate, behavioral_risk_score, packet_rate, failed_login_count):
    X_single = df[feature_columns].iloc[[0]].copy()

    if "byte_rate" in X_single.columns:
        X_single.loc[:, "byte_rate"] = byte_rate

    if "behavioral_risk_score" in X_single.columns:
        X_single.loc[:, "behavioral_risk_score"] = behavioral_risk_score

    if "packet_rate" in X_single.columns:
        X_single.loc[:, "packet_rate"] = packet_rate

    if "failed_login_count" in X_single.columns:
        X_single.loc[:, "failed_login_count"] = failed_login_count

    probabilities = model.predict_proba(X_single)[0]
    return probabilities

# Class prediction
def process_predictions(probabilities):
    classes = label_encoder.classes_
    breakdown = []
    for cls, prob in zip(classes, probabilities):
        breakdown.append({
                "class_name": cls,
                "percentage_str": f"{prob:.2%}",
                "raw_value": float(prob),
                "progress_percent": float(prob) * 100,
            })
    return {"final_prediction": classes[probabilities.argmax()], "breakdown": breakdown,}

@app.route("/", methods=["GET", "POST"])

def index():
    active_tab = "existing"
    error_message = None
    prediction_results = None

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "existing":
            active_tab = "existing"
            event_id = request.form.get("event_id", "").strip()
            probabilities = predict_threat_probability(event_id)

            if probabilities is None:
                error_message = (f"Event ID '{event_id}' not found.")
            else:
                prediction_results = process_predictions(probabilities)


        elif form_type == "new":
            active_tab = "new"
            try:
                byte_rate = float(request.form.get("byte_rate", 10000))
                behavioral_risk_score = float(request.form.get("behavioral_risk_score", 0.70))
                packet_rate = float(request.form.get("packet_rate", 70))
                failed_login_count = float(request.form.get("failed_login_count", 4))

                probabilities = (predict_raw_threat_probability(
                        byte_rate=byte_rate,
                        behavioral_risk_score=behavioral_risk_score,
                        packet_rate=packet_rate,
                        failed_login_count=failed_login_count))
                prediction_results = process_predictions(probabilities)

            except Exception as e:
                error_message = str(e)

    return render_template("index.html",
        active_tab=active_tab, error_message=error_message, results=prediction_results)

if __name__ == "__main__":
    app.run(debug=True)