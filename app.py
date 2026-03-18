# import streamlit as st
# import pandas as pd
# import seaborn as sns

# st.title("Titanic Dataset Analysis ")
# df = sns.load_dataset('titanic')

# st.dataframe(df.head())

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)

# ── Load & Preprocess Data ───────────────────
@st.cache_data
def load_data():
    df = sns.load_dataset('titanic')
    df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()
    df.dropna(inplace=True)
    return df

@st.cache_resource
def train_and_save_models(df):
    X = df.drop('survived', axis=1)
    y = df['survived']

    categorical_cols = ['sex', 'embarked']
    numerical_cols   = ['pclass', 'age', 'sibsp', 'parch', 'fare']

    preprocessor = ColumnTransformer(transformers=[
        ('ohe',    OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ('scaler', StandardScaler(),                                  numerical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p  = preprocessor.transform(X_test)

    # save everything in the same folder as app.py
    joblib.dump(preprocessor, 'preprocessor.pkl')

    models = {
        'DecisionTree' : DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForest' : RandomForestClassifier(n_estimators=100, random_state=42),
        'NaiveBayes'   : GaussianNB(),
        'SVC'          : SVC(kernel='rbf', probability=True, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train_p, y_train)
        joblib.dump(model, f'{name}.pkl')

    return X_test_p, y_test

# ════════════════════════════════════════════
st.title("🚢 Titanic — EDA & ML Classification")
st.markdown("Decision Tree · Random Forest · Naive Bayes · SVC")

df = load_data()

# ── SECTION 1: RAW DATA ──────────────────────
st.header("1. Dataset")
st.dataframe(df.head(10))
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── SECTION 2: EDA ───────────────────────────
st.header("2. Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival Count")
    fig, ax = plt.subplots()
    df['survived'].value_counts().plot(kind='bar', ax=ax, color=['#e74c3c','#2ecc71'])
    ax.set_xticklabels(['Not Survived', 'Survived'], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Survival by Sex")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sex', hue='survived', ax=ax,
                  palette={0:'#e74c3c', 1:'#2ecc71'})
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df[df['survived']==1]['age'], bins=20, alpha=0.6, label='Survived', color='#2ecc71')
    ax.hist(df[df['survived']==0]['age'], bins=20, alpha=0.6, label='Not Survived', color='#e74c3c')
    ax.legend()
    ax.set_xlabel("Age")
    st.pyplot(fig)

with col4:
    st.subheader("Survival by Pclass")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='pclass', hue='survived', ax=ax,
                  palette={0:'#e74c3c', 1:'#2ecc71'})
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
st.pyplot(fig)

# ── SECTION 3: TRAIN MODELS ──────────────────
st.header("3. Model Training")

X_test_p, y_test = train_and_save_models(df)
st.success("All 4 models trained and saved as .pkl files alongside app.py")

# ── SECTION 4: METRICS & CONFUSION MATRICES ──
st.header("4. Confusion Matrix & Metrics")

model_names = ['DecisionTree', 'RandomForest', 'NaiveBayes', 'SVC']
results = {}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, name in enumerate(model_names):
    model  = joblib.load(f'{name}.pkl')
    y_pred = model.predict(X_test_p)

    p  = precision_score(y_test, y_pred)
    r  = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'Precision': p, 'Recall': r, 'F1': f1}

    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Not Survived', 'Survived'])
    disp.plot(ax=axes[i], colorbar=False)
    axes[i].set_title(f'{name}\nP={p:.2f}  R={r:.2f}  F1={f1:.2f}')

plt.tight_layout()
st.pyplot(fig)

st.subheader("Metrics Summary")
results_df = pd.DataFrame(results).T.round(3)
st.dataframe(results_df)

# ── SECTION 5: USER PREDICTION ───────────────
st.header("5. Predict a Passenger")

col_a, col_b, col_c = st.columns(3)

with col_a:
    pclass   = st.selectbox("Passenger Class", [1, 2, 3])
    sex      = st.selectbox("Sex", ['male', 'female'])
    age      = st.slider("Age", 1, 80, 25)

with col_b:
    sibsp    = st.number_input("Siblings/Spouses aboard", 0, 8, 0)
    parch    = st.number_input("Parents/Children aboard", 0, 6, 0)

with col_c:
    fare     = st.number_input("Fare paid", 0.0, 520.0, 32.0)
    embarked = st.selectbox("Embarked", ['S', 'C', 'Q'])

if st.button("Predict Survival"):
    preprocessor = joblib.load('preprocessor.pkl')

    sample = pd.DataFrame([{
        'pclass': pclass, 'sex': sex, 'age': age,
        'sibsp': sibsp, 'parch': parch,
        'fare': fare, 'embarked': embarked
    }])

    sample_processed = preprocessor.transform(sample)

    st.subheader("Results")
    for name in model_names:
        model  = joblib.load(f'{name}.pkl')
        pred   = model.predict(sample_processed)[0]
        prob   = model.predict_proba(sample_processed)[0]
        label  = "✅ SURVIVED" if pred == 1 else "❌ Not Survived"
        col_name, col_result = st.columns([1, 3])
        with col_name:
            st.write(f"**{name}**")
        with col_result:
            st.write(f"{label} — probability: Not={prob[0]:.2f}, Survived={prob[1]:.2f}")
         