# 🩺 Healthcare AI Diagnostic & Grounded Knowledge Assistant

An end-to-end Healthcare AI prototype developed for an **AI/ML Technical Assessment**. The system unifies a leak-free tabular machine learning classification pipeline with a Retrieval-Augmented Generation (RAG) clinical question-answering assistant, wrapped inside an interactive multi-tab **Streamlit** web application.

---

> [!WARNING]
> ### Clinical Disclaimer
> **Demonstration and Assessment Prototype Only:** This software is developed solely for technical demonstration and recruitment evaluation. It is **not** an FDA-cleared diagnostic device, clinical decision support system, or medical diagnostic tool. It must never be used as a substitute for professional medical advice, clinical diagnosis, or patient management from a licensed physician or healthcare provider.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                      Streamlit Web Interface (app.py)                   │
├──────────────────────────┬─────────────────────────┬───────────────────┤
│ Tab 1: Risk & AI Expl.   │ Tab 2: Knowledge Assist │ Tab 3: Evaluation │
└────────────┬─────────────┴────────────┬────────────┴───────────────────┘
             │                          │
             ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────────────────┐
│ Tabular ML Pipeline      │  │ Grounded Healthcare RAG Engine           │
│ (src/train_ml.py)        │  │ (src/rag_engine.py & src/rag_indexer.py) │
├──────────────────────────┤  ├──────────────────────────────────────────┤
│ • Physiological 0s -> NaN│  │ • Overlapping Chunker (450 chars/80 ovlp)│
│ • Median Imputation      │  │ • Embeddings: all-MiniLM-L6-v2 (384-dim) │
│ • Standard Scaling       │  │ • Persistent FAISS FlatIP Vector Index   │
│ • Stratified 80/20 Split │  │ • Cosine Distance Guardrail (< 0.35 Ref) │
│ • Random Forest (Best)   │  │ • Strict Zero-Hallucination Grounding    │
│ • Output: Calibrated Prob│  │ • Multi-Provider (Gemini / OpenAI / Det.)│
└──────────────────────────┘  └──────────────────────────────────────────┘
```

---

## 🚀 Key Features

1. **Leak-Free Machine Learning Pipeline (`src/data_prep.py` & `src/train_ml.py`)**:
   - Accurately identifies and replaces physiologically impossible zeros in biological measurements (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with `NaN`.
   - Embeds `SimpleImputer(strategy='median')` and `StandardScaler()` strictly inside `scikit-learn.pipeline.Pipeline` objects fitted on training folds only to guarantee zero data leakage.
   - Evaluates multiple classifiers: **Logistic Regression** (interpretable linear baseline), **Random Forest** (non-linear ensemble trees), and **XGBoost** (gradient boosted trees).
   - Generates automated metrics reports (`reports/metrics.json`) recording **Accuracy, Precision, Recall, Specificity, F1-Score, and ROC-AUC**.
   - Exports the best-performing pipeline to `models/best_model.joblib`.
   - Exposes a reusable inference function: `predict_patient_risk(patient_data: dict) -> dict`.

2. **Grounded Healthcare RAG Pipeline (`src/rag_indexer.py` & `src/rag_engine.py`)**:
   - Ingests structured, evidence-based reference guidelines covering diabetes etiology, hypertension, lifestyle interventions (nutrition/exercise), HbA1c diagnostic cutoffs, medication adherence, preventive screening schedules, and acute red-flag emergencies.
   - Generates 384-dimensional dense semantic vectors using `sentence-transformers/all-MiniLM-L6-v2`.
   - Persists a high-performance **FAISS** vector store (`vector_store/index.faiss`) along with structured metadata.
   - **Anti-Hallucination Guardrails:** Enforces a strict similarity threshold (`SIMILARITY_THRESHOLD = 0.35`). Queries below the confidence cutoff (e.g., *"What is the cure for Alzheimer's?"*) trigger an immediate standard refusal:
     > *"I cannot answer this based on the provided healthcare documents."*
   - Cites referenced medical documents explicitly and supports **Gemini API**, **OpenAI API**, or an offline deterministic fallback engine.

3. **Multi-Tab Streamlit Web Application (`app.py`)**:
   - **Tab 1: Patient Risk & AI Explanation:** Interactive biometric input form with quick-fill sample presets (Low Risk, Prediabetic, High Risk). Outputs calibrated risk probability, risk category badge, identified clinical risk flags, and an automated tailored RAG explanation citing guidelines.
   - **Tab 2: Healthcare Knowledge Assistant:** Interactive Q&A chat interface with pre-loaded sample question buttons demonstrating both in-domain answers with source citations and out-of-scope refusal guardrails.
   - **Tab 3: Model Evaluation & Architecture:** Interactive metrics comparison table, visual performance charts, selection rationale, and pipeline architecture diagram.

---

## 📊 Model Comparison Benchmarks & Selection Rationale

The models were evaluated on the test set using a stratified 80/20 train/test split of the Pima Indians Diabetes dataset:

| Model Architecture | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | Composite Clinical Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 73.38% | 60.32% | 70.37% | 75.00% | 0.6496 | 0.8126 | 0.7636 |
| 🏆 **Random Forest (Selected)** | **73.38%** | **59.70%** | **74.07%** | **73.00%** | **0.6612** | **0.8204** | **0.7845** |
| **XGBoost Classifier** | 74.03% | 60.94% | 72.22% | 75.00% | 0.6610 | 0.8231 | 0.7777 |

### Why Random Forest Was Selected:
In clinical screening applications, **false negatives** (failing to identify an individual with impending or undiagnosed diabetes) present significantly greater medical danger than false positives. A false positive triggers a safe confirmatory laboratory test (such as a formal venous HbA1c or oral glucose tolerance test), whereas a false negative allows disease progression toward irreversible microvascular complications (retinopathy, nephropathy, neuropathy).

**Random Forest** achieved the **highest Recall (74.07%)** among all candidate architectures while maintaining a competitive **ROC-AUC of 0.8204**, maximizing the composite clinical screening score ($0.55 \times \text{ROC-AUC} + 0.45 \times \text{Recall} = 0.7845$).

---

## 📂 Repository Structure

```
.
├── .env.example                     # Sample environment variable template
├── .gitignore                       # Git exclusion rules (venv, models, caches)
├── requirements.txt                 # Pinned project dependencies
├── config.py                        # Centralized Pydantic configuration & paths
├── app.py                           # Multi-tab Streamlit web application
├── data/
│   ├── raw/
│   │   └── diabetes.csv             # Raw Pima Indians Diabetes dataset
│   └── docs/                        # Public health factsheets & clinical guidelines
│       ├── 01_diabetes_symptoms_and_types.txt
│       ├── 02_hypertension_and_cardiovascular_risk.txt
│       ├── 03_lifestyle_modifications_nutrition_exercise.txt
│       ├── 04_hba1c_and_glycemic_targets.txt
│       ├── 05_red_flag_symptoms_emergency_consultation.txt
│       ├── 06_preventive_screening_guidelines.txt
│       └── 07_medication_adherence_and_monitoring.txt
├── models/
│   └── best_model.joblib            # Serialized best-performing ML pipeline
├── vector_store/
│   ├── index.faiss                  # Persistent FAISS index
│   ├── metadata.json                # Chunk metadata and source mappings
│   └── embeddings.npy               # Normalized dense vector store backup
├── reports/
│   └── metrics.json                 # Offline evaluation metrics comparison report
└── src/
    ├── __init__.py
    ├── data_prep.py                 # Dataset download, zero cleaning, split
    ├── train_ml.py                  # Training, benchmarking, serialization, inference
    ├── rag_indexer.py               # Document chunking, MiniLM encoding, FAISS indexer
    └── rag_engine.py                # Retrieval, guardrails, citation, LLM caller
```

---

## ⚙️ Installation & Setup Instructions

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone <repo-url>
cd D_Intern

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional for Live LLM)

```bash
cp .env.example .env
```

Edit `.env` if you wish to use a live LLM API:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
LLM_PROVIDER=gemini
```
*(Note: If no API key is provided, the application automatically runs in deterministic offline mode, ensuring 100% functionality without external API dependencies.)*

### 3. Run Preprocessing and Train ML Models

```bash
python src/data_prep.py
python src/train_ml.py
```
This downloads the Pima Indians dataset (if not cached), executes the leak-free pipeline, generates `reports/metrics.json`, and serializes `models/best_model.joblib`.

### 4. Build FAISS Vector Index

```bash
python src/rag_indexer.py
```
This indexes the medical guidelines in `data/docs/`, generates vector embeddings with `all-MiniLM-L6-v2`, and persists the FAISS index to `vector_store/index.faiss`.

### 5. Launch the Streamlit Web Application

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📚 External APIs, Libraries, and Pretrained Models

| Category | Component / Model | Purpose / License |
| :--- | :--- | :--- |
| **Tabular ML** | `scikit-learn` (v1.9+) | Imputation, feature scaling, model training, cross-validation |
| **Boosting** | `xgboost` (v3.4+) | Extreme gradient boosted decision tree classifier |
| **Vector Search** | `faiss-cpu` (v1.16+) | High-performance dense vector nearest-neighbor search |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional dense semantic embedding model (Apache 2.0) |
| **LLM Provider** | `google-genai` / Gemini 2.5 Flash | Grounded clinical explanation generation with system constraints |
| **Web UI** | `streamlit` (v1.67+) | Responsive multi-tab clinical intelligence user interface |
| **Config** | `pydantic-settings` | Strongly-typed environment validation |

---

## 🔬 Dataset Source

- **Dataset Name:** Pima Indians Diabetes Database
- **Origin:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Host / Source:** [Kaggle / UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) / [GitHub Raw Mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- **Features (8 Biometric Predictors):**
  1. `Pregnancies`: Number of pregnancies
  2. `Glucose`: 2-hour oral glucose tolerance test plasma glucose (mg/dL)
  3. `BloodPressure`: Diastolic blood pressure (mm Hg)
  4. `SkinThickness`: Triceps skin fold thickness (mm)
  5. `Insulin`: 2-Hour serum insulin (μU/mL)
  6. `BMI`: Body mass index (weight in kg / (height in m)²)
  7. `DiabetesPedigreeFunction`: Genetic diabetes family history score
  8. `Age`: Age in years
- **Target:** `Outcome` (0: Non-Diabetic, 1: Diabetic)

---

## 🛡️ System Limitations & Future Engineering Roadmap

1. **Dataset Demographic Bias & Sample Size:**
   - *Limitation:* The Pima dataset consists of 768 female patients of Pima Indian heritage collected in Arizona. Risk coefficients may not generalize to diverse global multi-ethnic populations or male patients.
   - *Roadmap:* Integrate larger, multi-ethnic clinical datasets (e.g., NHANES, UK Biobank, or MIMIC-IV clinical databases) and assess cross-demographic calibration drift.

2. **Static Medical Corpus vs. Live Biomedical Search:**
   - *Limitation:* The RAG vector index is currently built on static public health factsheets. Emerging medical trial publications or updated clinical guidelines require manual re-indexing.
   - *Roadmap:* Implement a live PubMed / MeSH API integration with continuous incremental indexing and temporal filtering for recent clinical trials.

3. **Clinical Calibration & Interpretability (SHAP / LIME):**
   - *Limitation:* While the pipeline calculates class probabilities, tree-based models can exhibit overconfident probability tails without isotonic regression or Platt scaling.
   - *Roadmap:* Incorporate `CalibratedClassifierCV` for true Bayesian posterior risk alignment, and embed interactive TreeSHAP waterfall charts directly into Tab 1 to display individual feature contribution forces.

4. **HIPAA, Privacy & Clinical Compliance:**
   - *Limitation:* This demonstration prototype stores input parameters locally in memory and is designed for unauthenticated single-tenant use.
   - *Roadmap:* Implement end-to-end data encryption in transit and at rest, zero-logging LLM data retention agreements (HIPAA Business Associate Agreements - BAA), role-based access control (RBAC), and automated de-identification / PHI redaction pipelines.

---

## 📄 License
This prototype is released under the **MIT License** for academic and technical evaluation purposes.
