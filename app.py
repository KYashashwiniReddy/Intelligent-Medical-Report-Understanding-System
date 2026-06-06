# ==========================================
# IMPORTS
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import kagglehub

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fpdf import FPDF

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Medical Report Understanding System",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        "medical_attention_model.h5"
    )

    with open(
        "tokenizer.pkl",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    with open(
        "label_encoder.pkl",
        "rb"
    ) as f:

        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


# ==========================================
# LOAD DATASET FROM KAGGLE HUB
# ==========================================

@st.cache_data
def load_dataset():

    path = kagglehub.dataset_download(
        "tboyle10/medicaltranscriptions"
    )

    csv_file = os.path.join(
        path,
        "mtsamples.csv"
    )

    df = pd.read_csv(
        csv_file
    )

    df = df.dropna(
        subset=[
            "transcription",
            "medical_specialty"
        ]
    )

    return df


# ==========================================
# CREATE VOCABULARY
# ==========================================

@st.cache_data
def create_vocabulary(df):

    text = " ".join(
        df["transcription"].astype(str)
    )

    words = text.lower().split()

    vocab = Counter(words)

    vocab_df = pd.DataFrame(
        vocab.items(),
        columns=[
            "Medical_Term",
            "Frequency"
        ]
    )

    vocab_df = vocab_df.sort_values(
        by="Frequency",
        ascending=False
    )

    return vocab_df


# ==========================================
# LOAD EVERYTHING
# ==========================================

model, tokenizer, label_encoder = (
    load_artifacts()
)

df = load_dataset()

vocab_df = create_vocabulary(df)

MAX_LEN = 300
# ==========================================
# SIDEBAR MENU
# ==========================================

with st.sidebar:

    st.title("🏥 Medical NLP")

    selected = option_menu(
        menu_title=None,

        options=[
            "Home",
            "Medical Report Analyzer",
            "Dataset Insights",
            "Positional Encoding"
        ],

        icons=[
            "house",
            "activity",
            "bar-chart",
            "grid"
        ],

        default_index=0
    )


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_specialty(report_text):

    sequence = tokenizer.texts_to_sequences(
        [report_text]
    )

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction
    )

    specialty = (
        label_encoder
        .inverse_transform(
            [predicted_index]
        )[0]
    )

    confidence = (
        np.max(prediction)
        * 100
    )

    top_indices = np.argsort(
        prediction[0]
    )[::-1][:5]

    top_predictions = []

    for idx in top_indices:

        label = (
            label_encoder
            .inverse_transform(
                [idx]
            )[0]
        )

        score = (
            prediction[0][idx]
            * 100
        )

        top_predictions.append(
            (
                label,
                score
            )
        )

    return (
        specialty,
        confidence,
        top_predictions
    )


# ==========================================
# IMPORTANT TERMS
# ==========================================

def get_important_terms(text):

    stop_words = {
        "the","and","is","of","to",
        "a","in","for","with","on",
        "at","this","that","was",
        "are","be","by","an"
    }

    words = text.lower().split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return Counter(
        words
    ).most_common(15)


# ==========================================
# POSITIONAL ENCODING
# ==========================================

def positional_encoding(
        position,
        d_model):

    angle_rads = (
        np.arange(position)[:, np.newaxis]
        /
        np.power(
            10000,
            (
                2 *
                (
                    np.arange(d_model)[
                        np.newaxis, :
                    ] // 2
                )
            ) / d_model
        )
    )

    angle_rads[:, 0::2] = np.sin(
        angle_rads[:, 0::2]
    )

    angle_rads[:, 1::2] = np.cos(
        angle_rads[:, 1::2]
    )

    return angle_rads


# ==========================================
# PDF REPORT
# ==========================================

def create_pdf(
        specialty,
        confidence):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        size=12
    )

    pdf.cell(
        200,
        10,
        txt="Medical Report Analysis",
        ln=True
    )

    pdf.ln(10)

    pdf.cell(
        200,
        10,
        txt=f"Predicted Specialty: {specialty}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Confidence Score: {confidence:.2f}%",
        ln=True
    )

    file_name = (
        "Medical_Report_Analysis.pdf"
    )

    pdf.output(
        file_name
    )

    return file_name


# ==========================================
# KPI CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Reports",
    len(df)
)

col2.metric(
    "Specialties",
    df["medical_specialty"].nunique()
)

col3.metric(
    "Vocabulary",
    len(vocab_df)
)

st.markdown("---")
# ==========================================
# HOME PAGE
# ==========================================

if selected == "Home":

    st.title(
        "🏥 Intelligent Medical Report Understanding System"
    )

    st.markdown(
        """
        Analyze doctor reports and predict the
        corresponding medical specialty using
        Deep Learning, Self-Attention and
        Explainable AI techniques.
        """
    )

    st.markdown("---")

    # ==========================
    # OVERVIEW CARDS
    # ==========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Medical Reports",
        len(df)
    )

    col2.metric(
        "Specialties",
        df["medical_specialty"].nunique()
    )

    col3.metric(
        "Vocabulary Size",
        len(vocab_df)
    )

    st.markdown("---")

    # ==========================
    # PROJECT WORKFLOW
    # ==========================

    st.subheader(
        "Project Workflow"
    )

    st.code(
        """
Medical Report
      ↓
Text Cleaning
      ↓
Tokenizer
      ↓
Embedding Layer
      ↓
Self-Attention Layer
      ↓
Dense Neural Network
      ↓
Medical Specialty Prediction
      ↓
Explainable AI Dashboard
        """
    )

    st.markdown("---")

    # ==========================
    # SPECIALTY DISTRIBUTION
    # ==========================

    st.subheader(
        "Top Medical Specialties"
    )

    specialty_counts = (
        df["medical_specialty"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    specialty_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Top 10 Medical Specialties"
    )

    ax.set_xlabel(
        "Medical Specialty"
    )

    ax.set_ylabel(
        "Number of Reports"
    )

    plt.xticks(
        rotation=45
    )

    st.pyplot(fig)

    st.markdown("---")

    # ==========================
    # TOP MEDICAL TERMS
    # ==========================

    st.subheader(
        "Most Frequent Medical Terms"
    )

    top_words = vocab_df.head(15)

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.bar(
        top_words["Medical_Term"],
        top_words["Frequency"]
    )

    plt.xticks(
        rotation=45
    )

    ax.set_title(
        "Top Medical Terms"
    )

    st.pyplot(fig)

    st.markdown("---")

    st.success(
        "Use the sidebar menu to analyze medical reports and explore dataset insights."
    )
    # ==========================================
# MEDICAL REPORT ANALYZER
# ==========================================

if selected == "Medical Report Analyzer":

    st.title(
        "🩺 Medical Report Analyzer"
    )

    st.markdown(
        """
        Paste a medical report below and click
        **Analyze Report**.
        """
    )

    report_text = st.text_area(
        "Enter Medical Report",
        height=250
    )

    analyze_button = st.button(
        "Analyze Report"
    )

    if analyze_button:

        if len(
            report_text.strip()
        ) == 0:

            st.warning(
                "Please enter a medical report."
            )

        else:

            specialty, confidence, top_preds = (
                predict_specialty(
                    report_text
                )
            )

            st.markdown("---")

            # ==========================
            # PREDICTION RESULT
            # ==========================

            col1, col2 = st.columns(2)

            with col1:

                st.success(
                    f"Predicted Specialty: {specialty}"
                )

            with col2:

                st.metric(
                    "Confidence Score",
                    f"{confidence:.2f}%"
                )

            st.markdown("---")

            # ==========================
            # TOP 5 PREDICTIONS
            # ==========================

            st.subheader(
                "Top 5 Predictions"
            )

            prediction_df = pd.DataFrame(
                top_preds,
                columns=[
                    "Specialty",
                    "Confidence"
                ]
            )

            prediction_df[
                "Confidence"
            ] = prediction_df[
                "Confidence"
            ].round(2)

            st.dataframe(
                prediction_df,
                use_container_width=True
            )

            # ==========================
            # CONFIDENCE CHART
            # ==========================

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.barh(
                prediction_df["Specialty"],
                prediction_df["Confidence"]
            )

            ax.set_title(
                "Top Predicted Specialties"
            )

            ax.set_xlabel(
                "Confidence (%)"
            )

            st.pyplot(fig)

            st.markdown("---")

            # ==========================
            # IMPORTANT TERMS
            # ==========================

            st.subheader(
                "Important Medical Terms"
            )

            important_terms = (
                get_important_terms(
                    report_text
                )
            )

            terms_df = pd.DataFrame(
                important_terms,
                columns=[
                    "Medical Term",
                    "Frequency"
                ]
            )

            st.dataframe(
                terms_df,
                use_container_width=True
            )

            # ==========================
            # TERM FREQUENCY CHART
            # ==========================

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.bar(
                terms_df[
                    "Medical Term"
                ][:10],
                terms_df[
                    "Frequency"
                ][:10]
            )

            plt.xticks(
                rotation=45
            )

            ax.set_title(
                "Most Important Terms"
            )

            st.pyplot(fig)

            st.markdown("---")

            # ==========================
            # PDF DOWNLOAD
            # ==========================

            st.subheader(
                "Download Analysis Report"
            )

            pdf_file = create_pdf(
                specialty,
                confidence
            )

            with open(
                pdf_file,
                "rb"
            ) as file:

                st.download_button(
                    label="📄 Download PDF",
                    data=file,
                    file_name="Medical_Report_Analysis.pdf",
                    mime="application/pdf"
                )

            st.success(
                "Analysis Completed Successfully"
            )
            # ==========================================
# DATASET INSIGHTS
# ==========================================

if selected == "Dataset Insights":

    st.title(
        "📊 Dataset Insights"
    )

    st.markdown("---")

    # ==========================
    # DATASET STATISTICS
    # ==========================

    st.subheader(
        "Dataset Statistics"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reports",
        len(df)
    )

    col2.metric(
        "Total Specialties",
        df["medical_specialty"].nunique()
    )

    col3.metric(
        "Vocabulary Size",
        len(vocab_df)
    )

    st.markdown("---")

    # ==========================
    # SPECIALTY DISTRIBUTION
    # ==========================

    st.subheader(
        "Specialty Distribution"
    )

    specialty_counts = (
        df["medical_specialty"]
        .value_counts()
        .head(15)
    )

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    specialty_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Top 15 Medical Specialties"
    )

    ax.set_xlabel(
        "Medical Specialty"
    )

    ax.set_ylabel(
        "Number of Reports"
    )

    plt.xticks(
        rotation=45
    )

    st.pyplot(fig)

    st.markdown("---")

    # ==========================
    # PIE CHART
    # ==========================

    st.subheader(
        "Specialty Share"
    )

    fig, ax = plt.subplots(
        figsize=(8,8)
    )

    specialty_counts.head(10).plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")

    st.pyplot(fig)

    st.markdown("---")

    # ==========================
    # TOP MEDICAL TERMS
    # ==========================

    st.subheader(
        "Top Medical Terms"
    )

    top_words = vocab_df.head(20)

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    ax.bar(
        top_words["Medical_Term"],
        top_words["Frequency"]
    )

    plt.xticks(
        rotation=45
    )

    ax.set_title(
        "Most Frequent Medical Terms"
    )

    st.pyplot(fig)

    st.markdown("---")

    # ==========================
    # DATA PREVIEW
    # ==========================

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(),
        use_container_width=True
    )


# ==========================================
# POSITIONAL ENCODING
# ==========================================

if selected == "Positional Encoding":

    st.title(
        "📍 Positional Encoding Visualization"
    )

    st.markdown(
        """
        Positional Encoding helps the model
        understand the order of words in a sequence.
        """
    )

    st.markdown("---")

    st.subheader(
        "Positional Encoding Heatmap"
    )

    pos_encoding = positional_encoding(
        100,
        128
    )

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    heatmap = ax.imshow(
        pos_encoding,
        aspect="auto",
        cmap="RdYlBu"
    )

    plt.colorbar(
        heatmap
    )

    ax.set_title(
        "Positional Encoding Matrix"
    )

    ax.set_xlabel(
        "Embedding Dimension"
    )

    ax.set_ylabel(
        "Token Position"
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader(
        "Encoding Explanation"
    )

    st.info(
        """
        • Each row represents a token position.

        • Each column represents an embedding dimension.

        • Sine and cosine functions create unique position vectors.

        • Self-Attention uses these vectors to understand word order.

        • Without positional encoding, token order would be lost.
        """
    )

    st.markdown("---")

    st.success(
        "Positional Encoding Visualization Generated Successfully"
    )