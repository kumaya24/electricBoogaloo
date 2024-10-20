import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import datetime

st.set_page_config(
    page_title = "AEP Safeties Dashboard",
    layout = "wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

df_reshaped = pd.read_csv("aep_data.csv")
comments = df_reshaped['PNT_ATRISKNOTES_TX'].tolist()

with st.sidebar:
    st.title('Types of Load')
    pnt_list = list(df_reshaped.PNT_NM.unique())[::-1]
    severity_list = ['HSIF', 'Capacity', 'PSIF', 'Success', 'Exposure', 'LSIF', 'Low Severity']

    dates = list(df_reshaped.DATETIME_DTM.unique())
    years = []
    for date in dates: 
        datestr = str(date)
        date_list = datestr.split('/')
        year = date_list[2][0:4]
        if year not in years:
            years.append(year)
    year_list = sorted(years)[::-1]
    selected_pnt = st.selectbox('Observation type', severity_list, index=len(severity_list) - 1)

    selected_year = st.selectbox('Year', year_list, index=len(year_list) - 1)

def calculate_high_level_incidents():
    # put model in later to determine
    return len(comments) - 55

col = st.columns((1.5, 4.5, 2), gap = 'medium')

with col[0]:
    st.markdown('Number of Instances')

    hlincidents = calculate_high_level_incidents()
    llincidents = len(comments) - hlincidents

    st.metric(label="High Risk Instances", value=str(hlincidents), delta="5")
    st.metric(label="Low Risk Instances", value=str(llincidents), delta="-5")
    

with col[1]:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_count = [0,0,0,0,0,0,0,0,0,0,0,0]
    for date in dates:
        datestr = str(date)
        date_list = datestr.split('/')
        year = date_list[2][0:4]
        month = int(date_list[0])
        if year == selected_year:
            month_count[month - 1] = month_count[month - 1] + 1
    chart_data = pd.DataFrame(
        {
            "x-val": months,
            "y-val": month_count
        }
    )
    
    
    st.bar_chart(chart_data, x="x-val", y="y-val")