# segmentation.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI
import uvicorn
import pickle
import sqlite3
import os
from datetime import datetime

# ========== Step 1: Load and Prepare Data ==========

def load_data():
    user_df = pd.read_csv("user-evobee.csv")
    order_df = pd.read_excel("Order-ID-evobee.xlsx")

    # Fix date parsing
    order_df["TransactionDate"] = pd.to_datetime(order_df["TransactionDate"], errors='coerce')
    user_df["user_birthday"] = pd.to_datetime(user_df["user_birthday"], errors='coerce')

    # Normalize phone numbers to match
    user_df["user_phone"] = user_df["user_phone"].astype(str).str[-6:]
    order_df["Customer Phone"] = order_df["Customer Phone"].astype(str).str[-6:]

    # Merge on phone
    merged = pd.merge(order_df, user_df, left_on="Customer Phone", right_on="user_phone", how="left")

    # Clean up TransactionAmount
    merged["TransactionAmount"] = (
        merged["TransactionAmount"]
        .astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float)
    )

    return merged


# ========== Step 2: Calculate RFM Features ==========

def calculate_rfm(df):
    now = pd.to_datetime("today")

    rfm = df.groupby("UserID").agg({
        "TransactionDate": lambda x: (now - x.max()).days,
        "TransactionID": "count",
        "TransactionAmount": "sum",
        "user_firstname": "first"
    }).rename(columns={
        "TransactionDate": "Recency",
        "TransactionID": "Frequency",
        "TransactionAmount": "Monetary",
        "user_firstname": "Firstname"
    })

    return rfm


# ========== Step 3: Preprocess and Cluster ==========

def preprocess_rfm(rfm):
    features = rfm[["Recency", "Frequency", "Monetary"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler

def train_kmeans(data, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(data)
    return model


# ========== Step 4: Save Outputs ==========

def save_all(model, scaler, rfm):
    with open("kmeans_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    conn = sqlite3.connect("segments.db")
    rfm.reset_index().to_sql("segments", conn, if_exists="replace", index=False)
    conn.close()


# ========== Step 5: Train Once on Startup ==========

if not os.path.exists("segments.db"):
    print("Training initial model...")
    data = load_data()
    rfm = calculate_rfm(data)
    scaled, scaler = preprocess_rfm(rfm)
    model = train_kmeans(scaled)
    rfm["Segment"] = model.labels_
    save_all(model, scaler, rfm)
    print("Model trained and saved.")


# ========== Step 6: FastAPI Server ==========

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Evobee Customer Segmentation API"}


@app.get("/segment/{user_id}")
def get_segment(user_id: int):
    conn = sqlite3.connect("segments.db")
    cur = conn.cursor()
    result = cur.execute("SELECT Segment, Firstname FROM segments WHERE UserID = ?", (user_id,)).fetchone()
    conn.close()

    if result:
        segment, firstname = result
        return {
            "UserID": user_id,
            "Firstname": firstname,
            "Segment": int(segment)
        }

    return {"error": "User not found"}


@app.get("/summary")
def segment_summary():
    conn = sqlite3.connect("segments.db")
    df = pd.read_sql("SELECT * FROM segments", conn)
    conn.close()

    summary = df.groupby("Segment").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "UserID": "count"
    }).rename(columns={"UserID": "Count"}).reset_index()

    # Sample names per segment (optional, up to 5)
    samples = df.groupby("Segment")["Firstname"].apply(lambda x: list(x.dropna().unique())[:5]).reset_index()
    summary = summary.merge(samples, on="Segment")
    summary.rename(columns={"Firstname": "SampleFirstnames"}, inplace=True)

    return summary.to_dict(orient="records")


@app.post("/retrain")
def retrain():
    print("Retraining model...")
    data = load_data()
    rfm = calculate_rfm(data)
    scaled, scaler = preprocess_rfm(rfm)
    model = train_kmeans(scaled)
    rfm["Segment"] = model.labels_
    save_all(model, scaler, rfm)
    return {"message": "Model retrained and segments updated."}





# http://127.0.0.1:8000/  To check if API is Running

# http://127.0.0.1:8000/segment/userID to get the individual user data

# http://127.0.0.1:8000/summary to get the summary of segments

#  http://127.0.0.1:8000/docs FastAPI built in docs



# Segment	   Description	               RFM Characteristics

#  0	   Possibly Average Customers	   Medium recency, frequency, and spending

#  1	     High-Value Loyal Users	       Very low recency, high frequency, high monetary

#  2	     New/One-time User	           Possibly 1 transaction, low spend, recent activity

#  3	    At-Risk or Lost Customers	   High recency (inactive), low frequency and monetary
