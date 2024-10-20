import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(
    page_title = "AEP Safeties Dashboard",
    layout = "wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

df_reshaped = pd.read_csv("labeled_data.csv")
df_size = len(df_reshaped)
keydata = []
for i in range(df_size):
    row = df_reshaped.iloc[i]
    datestr = str(row.iloc[1])
    date_list = datestr.split('/')
    year = date_list[2][0:4]
    month = date_list[0]
    pnt_nm = row.iloc[2]
    comment = row.iloc[4]
    high_value = row.iloc[6]
    keydata.append((month, year, pnt_nm, comment, high_value))

comments = df_reshaped['PNT_ATRISKNOTES_TX'].tolist()

with st.sidebar:
    st.title('Types of Load')
    pnt_list = list(df_reshaped.PNT_NM.unique())[::-1]
    severity_list = ['HSIF', 'Capacity', 'PSIF', 'Success', 'Exposure', 'LSIF', 'Low Severity']

    dates = list(df_reshaped.DATETIME_DTM.unique())
    years = ['All']
    for date in dates: 
        datestr = str(date)
        date_list = datestr.split('/')
        year = date_list[2][0:4]
        if year not in years:
            years.append(year)
    year_list = sorted(years)[::-1]
    selected_pnt = st.selectbox('Observation type', severity_list, index=len(severity_list) - 1)

    selected_year = st.selectbox('Year', year_list, index=len(year_list) - 1)

def get_year_data(year, data):
    yeardata = []
    for month_yr, year_yr, pnt_nm_yr, comment_yr, hv_yr in data:
        if year_yr == str(year):
            yeardata.append((month_yr, year_yr, pnt_nm_yr, comment_yr, hv_yr))
    return yeardata

def calculate_high_level_instances(data):
    # put model in later to determine
    count = 0
    for month_yr, year_yr, pnt_nm_yr, comment_yr, hv_yr in data:
        if hv_yr == 'yes':
            count = count + 1
    return count

col = st.columns((1.5, 4.5, 2), gap = 'medium')

with col[0]:
    st.markdown('Number of Instances')
    if selected_year == 'All':
        fullyeardata = keydata
    else:
        fullyeardata = get_year_data(selected_year, keydata)
    
    hlinst = calculate_high_level_instances(fullyeardata)
    llinst = len(fullyeardata) - hlinst

    st.metric(label="High Risk Instances", value=str(hlinst), delta="5")
    st.metric(label="Low Risk Instances", value=str(llinst), delta="-5")
    

with col[1]:
    cloudtext = ''
        

    if selected_year == 'All':
        st.markdown('Number of Total High Value Instances')
        year_count_dict = {}
        for year in year_list:
            if year != 'All':
                year_count_dict[year] = 0

        for mon, yr, key, com, hv in keydata:
            year_count_dict[yr] += 1

        chart_data = pd.DataFrame(
            {
                "Year": year_count_dict.keys(),
                "Instances": year_count_dict.values()
            }
        )
        st.bar_chart(chart_data, x="Year", y="Instances")
    else:
        st.markdown('Number of High Value Instances in ' + selected_year)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_count = [0,0,0,0,0,0,0,0,0,0,0,0]

        for mon, yr, key, com, hv in fullyeardata:
            if hv == 'yes':
                month_count[int(mon) - 1] = month_count[int(mon) - 1] + 1

        chart_data = pd.DataFrame(
            {
                "Month": months, 
                "Instances": month_count,
            }
        )
        st.bar_chart(chart_data, x="Month", y="Instances")

st.dataframe(df_reshaped)