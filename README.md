# Smart Loan Recovery System

> An AI-powered system for **loan risk segmentation**, **borrower profiling**, and **recovery strategy optimization**, built with **Python**, **Machine Learning**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-app-success?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn)
![Plotly](https://img.shields.io/badge/Plotly-visualization-blueviolet?logo=plotly)

---

## Problem Statement

Banks and financial institutions often struggle with **inefficient loan recovery** processes and **rising default rates**.  
Manually identifying high-risk borrowers and deciding the best recovery action is time-consuming and inconsistent.

This project automates that process using **Machine Learning**:
- Segments borrowers based on financial and behavioral factors using **K-Means clustering**.  
- Predicts **loan default risk** using a **Random Forest classifier**.  
- Generates **personalized recovery strategies**- ranging from reminders to legal escalation.  
- Presents all insights via an **interactive Streamlit dashboard**.

---

##  Tech Stack

| Category | Tools |
|-----------|------------------|
| **Language** | Python |
| **IDE** | Visual Studio Code |
| **Framework / UI** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (KMeans, RandomForestClassifier) |
| **Visualization** | Plotly, Plotly Express |
| **Version Control** | Git + GitHub |
| **Deployment** | Streamlit Cloud |

---

## Demo

🔗 **Live App:** [Smart Loan Recovery System](https://loansystempy-gpnsbgqrnipcli4eqymwwd.streamlit.app/)  

---

## How to Run Locally

### 1) Clone the repository
```bash
git clone https://github.com/aparajita1721/Smart_loan_recovery_system.git
cd Smart_loan_recovery_system
```
### 2) Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 3) Install dependencies
```bash
pip install -r requirements.txt
```
 ### 4) Run the app
 ```bash
streamlit run loan_system.py
```


