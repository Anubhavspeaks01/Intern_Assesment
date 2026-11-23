"""
Healthcare AI Prototype - Streamlit Web Application
Unifies Tabular ML Classification with Grounded Retrieval-Augmented Generation (RAG).
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from config import settings
from src.train_ml import predict_patient_risk, load_inference_pipeline
from src.rag_engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Healthcare AI Diagnostic & Knowledge Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        color: #92400E;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        font-size: 0.95rem;
        font-weight: 600;
        border-radius: 9999px;
        color: white;
    }
    .source-tag {
        display: inline-block;
        background-color: #EEF2F6;
        border: 1px solid #CBD5E1;
        border-radius: 4px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing RAG Engine & FAISS Vector Store...")
def get_rag_engine():
    """Cache RAG engine to avoid re-loading embeddings and index on every rerun."""
    return RAGEngine()


# Top Banner & Disclaimer
st.markdown('<div class="main-header">🩺 Healthcare AI Intelligence System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Unified Predictive Clinical Risk Classification & Grounded Healthcare Knowledge Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="disclaimer-banner">
        ⚠️ <strong>Clinical Assessment Prototype & Demonstration Only:</strong> This application is developed for technical assessment and educational evaluation. It is <strong>not</strong> an FDA-cleared diagnostic medical device and should never replace clinical judgment, diagnosis, or care from a qualified healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar System Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-heart.png", width=64)
    st.subheader("System Architecture")
    st.markdown(
        """
        - **ML Classifier:** Scikit-Learn / XGBoost leak-free pipeline with median imputation and feature scaling.
        - **Vector Store:** FAISS Flat Inner Product index with 384-dimensional dense vectors.
        - **Embeddings:** `all-MiniLM-L6-v2`
        - **Grounding Rule:** Zero-hallucination threshold with automatic out-of-scope refusal.
        """
    )
    st.divider()
    st.subheader("Configuration Status")
    st.write(f"**LLM Provider:** `{settings.LLM_PROVIDER}`")
    has_gemini = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY != "your_gemini_api_key_here")
    has_openai = bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "your_openai_api_key_here")
    st.write(f"**Gemini API Key:** {'✅ Configured' if has_gemini else '⚠️ Not Set (Fallback Mode Active)'}")
    st.write(f"**OpenAI API Key:** {'✅ Configured' if has_openai else '⚠️ Not Set'}")
    st.write(f"**Similarity Refusal Threshold:** `{settings.SIMILARITY_THRESHOLD}`")


# Initialize Navigation Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Tab 1: Patient Risk & AI Explanation",
    "💬 Tab 2: Healthcare Knowledge Assistant",
    "📈 Tab 3: Model Evaluation & Architecture",
])


# ==============================================================================
# TAB 1: PATIENT RISK PREDICTION & EXPLANATION
# ==============================================================================
with tab1:
    st.markdown("### Patient Clinical Risk Assessment")
    st.markdown(
        "Enter clinical biometric measurements to compute the calibrated diabetes risk probability "
        "and automatically generate a grounded medical explanation citing relevant guidelines."
    )

    # Patient Presets for Quick Demonstration
    col_presets, _ = st.columns([3, 1])
    with col_presets:
        preset = st.radio(
            "Load Clinical Sample Preset:",
            options=["Manual Entry", "Low Risk Patient Profile", "Borderline / Prediabetic Profile", "High Risk Diabetic Profile"],
            horizontal=True,
        )

    # Preset Defaults
    if preset == "Low Risk Patient Profile":
        def_preg, def_gluc, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = (1, 88.0, 68.0, 20.0, 75.0, 22.4, 0.22, 24)
    elif preset == "Borderline / Prediabetic Profile":
        def_preg, def_gluc, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = (3, 118.0, 78.0, 28.0, 110.0, 28.5, 0.45, 42)
    elif preset == "High Risk Diabetic Profile":
        def_preg, def_gluc, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = (6, 172.0, 92.0, 36.0, 230.0, 35.8, 0.88, 54)
    else:
        def_preg, def_gluc, def_bp, def_skin, def_ins, def_bmi, def_dpf, def_age = (2, 110.0, 72.0, 25.0, 90.0, 27.0, 0.35, 35)

    with st.form("risk_assessment_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Metabolic & Glycemic Markers")
            glucose = st.number_input(
                "Fasting Plasma Glucose (mg/dL)",
                min_value=0.0,
                max_value=300.0,
                value=def_gluc,
                step=1.0,
                help="Reference: <100 mg/dL Normal, 100-125 Prediabetes, >=126 Diabetes. (0 flags as missing)",
            )
            insulin = st.number_input(
                "2-Hour Serum Insulin (μU/mL)",
                min_value=0.0,
                max_value=850.0,
                value=def_ins,
                step=1.0,
                help="Normal fasting insulin ~ 2-20 μU/mL; postprandial levels vary. (0 flags as missing)",
            )
            dpf = st.number_input(
                "Diabetes Pedigree Function (Genetic History Score)",
                min_value=0.05,
                max_value=2.50,
                value=def_dpf,
                step=0.01,
                help="Scores genetic influence based on family history of diabetes.",
            )
            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=def_preg,
                step=1,
            )

        with col2:
            st.markdown("##### Anthropometric & Hemodynamic Markers")
            bmi = st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=0.0,
                max_value=70.0,
                value=def_bmi,
                step=0.1,
                help="Reference: 18.5-24.9 Normal, 25.0-29.9 Overweight, >=30.0 Obese. (0 flags as missing)",
            )
            blood_pressure = st.number_input(
                "Diastolic Blood Pressure (mmHg)",
                min_value=0.0,
                max_value=180.0,
                value=def_bp,
                step=1.0,
                help="Normal < 80 mmHg, Stage 1 HTN 80-89, Stage 2 HTN >= 90. (0 flags as missing)",
            )
            skin_thickness = st.number_input(
                "Triceps Skinfold Thickness (mm)",
                min_value=0.0,
                max_value=99.0,
                value=def_skin,
                step=1.0,
                help="Surrogate marker for peripheral subcutaneous body fat. (0 flags as missing)",
            )
            age = st.number_input(
                "Age (Years)",
                min_value=18,
                max_value=110,
                value=def_age,
                step=1,
                help="Age >= 45 years is a major independent screening risk factor.",
            )

        submitted = st.form_submit_button("🔍 Calculate Risk Assessment & AI Explanation", use_container_width=True)

    if submitted or preset != "Manual Entry":
        patient_data = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }

        with st.spinner("Executing ML Inference Pipeline..."):
            risk_result = predict_patient_risk(patient_data)

        prob = risk_result["risk_percentage"]
        tier = risk_result["risk_tier"]
        color = risk_result["risk_color"]
        flags = risk_result["clinical_flags"]

        st.divider()
        st.subheader("Assessment Results")

        res_col1, res_col2, res_col3 = st.columns([1, 1, 1.5])
        with res_col1:
            st.metric(label="Calculated Diabetes Risk", value=f"{prob}%")
            st.caption("Derived from the best-performing validated ML pipeline.")
        with res_col2:
            st.markdown("**Assigned Risk Category:**")
            st.markdown(
                f'<span class="risk-badge" style="background-color: {color};">{tier}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Predicted Class: **{risk_result['predicted_label']}** (Threshold 0.50)")
        with res_col3:
            st.markdown("**Identified Clinical Risk Factors:**")
            if flags:
                for f in flags:
                    st.markdown(f"- ⚠️ {f}")
            else:
                st.markdown("✅ No critical abnormal biometric markers identified.")

        # Trigger Tailored RAG Explanation
        st.markdown("#### 📖 Tailored Clinical AI Explanation (Grounded in Guidelines)")
        with st.spinner("Generating personalized explanation from medical reference documents..."):
            try:
                rag_engine = get_rag_engine()
                explanation = rag_engine.explain_patient_risk(risk_result)
                st.info(explanation)
            except Exception as e:
                st.error(f"Error generating RAG explanation: {e}")


# ==============================================================================
# TAB 2: HEALTHCARE KNOWLEDGE ASSISTANT (RAG Q&A)
# ==============================================================================
with tab2:
    st.markdown("### Interactive Healthcare Knowledge Assistant")
    st.markdown(
        "Ask questions regarding diabetes symptoms, hypertension, HbA1c diagnostic cutoffs, "
        "and emergency red flags. The assistant is strictly grounded to avoid hallucinations and will "
        "refuse queries outside its verified medical knowledge base."
    )

    # Sample query buttons
    st.markdown("**Quick Test Sample Queries:**")
    btn_cols = st.columns(4)
    sample_selected = None
    with btn_cols[0]:
        if st.button("🩸 Early Symptoms", use_container_width=True):
            sample_selected = "What are the early warning symptoms of diabetes?"
    with btn_cols[1]:
        if st.button("🚨 Emergency Red Flags", use_container_width=True):
            sample_selected = "When should a patient seek emergency care or call 911 for diabetic complications?"
    with btn_cols[2]:
        if st.button("🥗 Diet & Exercise", use_container_width=True):
            sample_selected = "What are the recommended lifestyle modifications and diet for prediabetes?"
    with btn_cols[3]:
        if st.button("❌ Test Out-of-Scope Refusal", use_container_width=True):
            sample_selected = "What is the cure for Alzheimer's disease?"

    user_query = st.text_input(
        "Enter your healthcare question:",
        value=sample_selected if sample_selected else "",
        placeholder="e.g., What are the normal and prediabetic HbA1c ranges?",
    )

    if st.button("Submit Question", type="primary") or sample_selected:
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching FAISS vector index and generating grounded response..."):
                try:
                    rag_engine = get_rag_engine()
                    res = rag_engine.ask(user_query)

                    st.markdown("---")
                    st.subheader("AI Response")

                    if res["refusal"]:
                        st.warning(f"🛡️ **Guardrail Triggered:** {res['answer']}")
                        st.caption(
                            f"Cosine similarity score ({res['max_similarity']}) is below the strict confidence cutoff "
                            f"({settings.SIMILARITY_THRESHOLD}). Hallucination prevented."
                        )
                    else:
                        st.markdown(res["answer"])
                        st.markdown(f"**Relevance Confidence (Similarity):** `{res['max_similarity']:.4f}`")

                    # Sources list
                    if res["sources"]:
                        st.markdown("##### Referenced Medical Documents:")
                        for src in res["sources"]:
                            st.markdown(f'<span class="source-tag">📄 {src}</span>', unsafe_allow_html=True)

                    # Retrieved chunks expander
                    if res["retrieved_chunks"]:
                        with st.expander("🔍 View Retrieved Context Passages & Similarity Scores"):
                            for i, c in enumerate(res["retrieved_chunks"]):
                                st.markdown(f"**Chunk {i+1} | Source:** `{c['source_name']}` | **Similarity:** `{c.get('similarity_score', 0.0):.4f}`")
                                st.code(c["text"], language="text")

                except Exception as e:
                    st.error(f"Error querying RAG engine: {e}")


# ==============================================================================
# TAB 3: MODEL EVALUATION & ARCHITECTURE
# ==============================================================================
with tab3:
    st.markdown("### Machine Learning Model Evaluation & Benchmark Comparison")
    st.markdown(
        "Comprehensive offline performance benchmarking of candidate classification models "
        "trained on the Pima Indians Diabetes cohort."
    )

    metrics_file = settings.METRICS_PATH
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics_data = json.load(f)

        best_model = metrics_data.get("best_model", "N/A")
        selection_rationale = metrics_data.get("selection_rationale", "")
        models_dict = metrics_data.get("models", {})

        st.success(f"🏆 **Selected Best Model:** **{best_model}**")
        st.markdown(f"**Clinical Selection Rationale:** {selection_rationale}")

        # Build comparison table
        table_rows = []
        for name, m in models_dict.items():
            table_rows.append({
                "Model Architecture": name,
                "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                "Precision": f"{m['precision'] * 100:.2f}%",
                "Recall (Sensitivity)": f"{m['recall'] * 100:.2f}%",
                "Specificity": f"{m.get('specificity', 0.0) * 100:.2f}%",
                "F1-Score": f"{m['f1_score']:.4f}",
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "Clinical Score": f"{m.get('composite_clinical_score', 0.0):.4f}",
            })
        df_metrics = pd.DataFrame(table_rows)
        st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        # Visual Comparison Bar Chart
        chart_data = []
        for name, m in models_dict.items():
            chart_data.append({"Model": name, "Metric": "Recall", "Score": m["recall"]})
            chart_data.append({"Model": name, "Metric": "ROC-AUC", "Score": m["roc_auc"]})
            chart_data.append({"Model": name, "Metric": "Accuracy", "Score": m["accuracy"]})
        chart_df = pd.DataFrame(chart_data)

        st.markdown("#### Model Performance Metrics Comparison")
        st.bar_chart(chart_df, x="Model", y="Score", color="Metric")

    else:
        st.warning("Metrics file not found. Run `python src/train_ml.py` to generate benchmarking results.")

    st.divider()
    st.subheader("System Architecture & Data Flow")
    st.markdown(
        """
        ```
        [Raw Clinical Data] (Pima Indians Dataset)
                 │
                 ▼
        [Data Preprocessing Pipeline] ──> Handles physiological zeros as NaN
                 │                     ──> Median Imputation (leak-free inside folds)
                 │                     ──> Feature Standardization (StandardScaler)
                 ▼
        [Stratified 80/20 Train/Test Split]
                 │
                 ├──> [Logistic Regression] (Linear Interpretable Baseline)
                 ├──> [Random Forest]       (Ensemble Decision Trees)
                 └──> [XGBoost]             (Gradient Boosted Trees)
                             │
                             ▼
                 [Clinical Evaluation & Model Serialization]
                 (Balancing Recall & ROC-AUC -> models/best_model.joblib)

        [Medical Factsheets (WHO/CDC/ADA)]
                 │
                 ▼
        [Text Cleaning & Overlapping Chunker] (450 chars / 80 overlap)
                 │
                 ▼
        [Sentence Transformers (all-MiniLM-L6-v2)]
                 │
                 ▼
        [Persistent FAISS Vector Store] (Flat Inner Product / Cosine Distance)
                 │
                 ▼
        [RAG Engine Guardrails] ──> Cosine Similarity Filter (< 0.35 Refusal)
                                ──> Strict In-Context Grounding
                                ──> Automatic Source Citation
        ```
        """
    )
