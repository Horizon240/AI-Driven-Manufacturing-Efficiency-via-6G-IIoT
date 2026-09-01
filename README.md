# 🏭 Smart Factory 6G Dashboard: AI-Driven Manufacturing Efficiency

## 🎯 Strategic Objective
As industrial sectors transition toward autonomous smart factories, the reliance on high-frequency Industrial IoT (IIoT) sensors and Ultra-Reliable Low-Latency Communication (URLLC) via 6G networks has introduced unprecedented data complexity. Traditional reactive dashboards fail to capture the immediate, non-linear impacts of network latency and sensor deviations on production output. This project provides a real-time predictive classification system to secure manufacturing efficiency and prevent cascading production losses.

---

## 🏗️ Technical Solution & Architecture
This repository contains an end-to-end machine learning pipeline that shifts factory monitoring from diagnostic (looking backward) to predictive (looking forward).

### 1. Data Integration
The system ingests high-fidelity 6G telemetry (latency, packet loss) alongside physical machine metrics (temperature, vibration, power consumption) to create a unified operational profile.

### 2. Predictive Engine
An advanced XGBoost classification model, optimized with Synthetic Minority Over-sampling Technique (SMOTE), trained on multivariate data. The model achieves 99.7% accuracy in classifying the live efficiency state of the factory floor (Optimal, Medium, or Degraded).

### 3. Live Control Room
The predictive engine is deployed via an interactive Streamlit application, allowing operators to monitor live telemetry, filter historical degradation patterns, and conduct "What-If" threshold testing in real-time.

---

## ⚙️ Key Operational Capabilities

* **Automated State Detection:** Instantly alerts operators when production efficiency degrades due to interconnected hardware or network stress.
* **Root-Cause Explainability:** Integrates SHAP (SHapley Additive exPlanations) to break the "black box" of machine learning. When efficiency drops, the system explicitly ranks the driving factors (e.g., a 40ms spike in 6G latency vs. a 15-degree temperature anomaly), enabling immediate, targeted maintenance.
* **Network-Aware Analytics:** Uniquely treats 6G connectivity health as a primary variable, ensuring that data synchronization bottlenecks are caught before they impact physical manufacturing quality.

---

## 💻 Tech Stack

* **Core:** Python 3.13
* **Data Processing:** Pandas, NumPy, Scikit-learn, Imbalanced-learn (SMOTE)
* **Machine Learning:** XGBoost
* **Explainable AI:** SHAP
* **Frontend/Dashboard:** Streamlit, Plotly, Matplotlib

---

## 🚀 Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/smart-factory-6g-dashboard.git](https://github.com/yourusername/smart-factory-6g-dashboard.git)
cd smart-factory-6g-dashboard```
Installation & Setup
Clone the repository:

Bash
git clone https://github.com/yourusername/smart-factory-6g-dashboard.git
cd smart-factory-6g-dashboard
Install required dependencies:

Bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost shap streamlit plotly matplotlib
Run the model pipeline (generates .pkl files):

Bash
python pipeline.py
Launch the Streamlit Dashboard:

Bash
python -m streamlit run app.py
Usage Guide
Historical Filters: Use the sidebar to select specific Machine IDs and filter out high-latency network events to view historical production output and scatter profiles.

Live What-If Analysis: Adjust the Temperature, Vibration, 6G Latency, and Power sliders in the sidebar. The main dashboard will instantly recalculate the predicted efficiency state and update the confidence gauge.

Root Cause Analysis: Scroll to the bottom panel to view the SHAP waterfall chart. This chart updates dynamically with the sidebar sliders, explicitly showing which metrics are currently driving the factory toward or away from peak efficiency.
