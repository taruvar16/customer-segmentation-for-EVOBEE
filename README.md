# customer-segmentation-for-EVOBEE

# Evobee Customer Segmentation

This project performs customer segmentation using RFM (Recency, Frequency, Monetary) analysis and KMeans clustering. It integrates with a FastAPI server for real-time segment lookup.

## 🚀 Features

- Loads transaction and user data
- Computes RFM scores
- Clusters users into segments
- Saves results to SQLite and Pickle
- Provides FastAPI endpoint to fetch user segment

## 🧰 Tech Stack

- **Language**: Python
- **Libraries**: pandas, scikit-learn, FastAPI, numpy, uvicorn
- **Persistence**: SQLite, Pickle
- **Serving**: FastAPI REST API

## 📂 Folder Structure
evobee-customer-segmentation/
├── evobee_segmentation/
│ └── segmentation.py # Core processing and FastAPI server
├── models/ # Saved models (kmeans_model.pkl, scaler.pkl)
├── README.md # Project description
├── requirements.txt # Dependencies
└── .gitignore # Ignore cache, db, pkl, etc.
