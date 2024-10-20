import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

csv = st.file_uploader("Upload a csv")

import llm

if csv is not None:
    if st.button("Upload"):
        input_csv = pd.read_csv(csv)
        # clear file
        open(csv.name, "w").close()

        # write to file
        input_csv.to_csv(csv.name, index=False)

        st.write("Uploaded!")
        with st.spinner('Running...'):
            llm.runModelOnCSV(csv.name)
        st.success("Done!")
        


