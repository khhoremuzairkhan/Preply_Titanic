import streamlit as st
import pandas as pd
import seaborn as sns

st.title("Titanic Dataset Analysis ")
df = sns.load_dataset('titanic')

st.dataframe(df.head())