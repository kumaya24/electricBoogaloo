import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import datetime
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random

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

# Function to calculate the percentage
def calculate_percentage(df_selected, df_all):
    if len(df_all) == 0:  # Handle empty dataset
        return 0.0
    percentage = (len(df_selected) / len(df_all)) * 100  # Calculate percentage
    return round(percentage, 2)  # Round to two decimals

# Function to create an enhanced donut chart
def make_donut(percentage, input_text):
    # Define colors for chart
    chart_colors = ['#E0E0E0', '#4A90E2']  # Blue and light gray for contrast

    # Data for the donut chart
    source = pd.DataFrame({
        "Topic": [input_text, 'Remaining'],
        "% value": [percentage, 100 - percentage]
    })

    # Create the main donut chart
    plot = alt.Chart(source).mark_arc(innerRadius=60, cornerRadius=30).encode(
        theta=alt.Theta("% value:Q"),
        color=alt.Color('Topic:N', scale=alt.Scale(range=chart_colors), legend=None)
    ).properties(width=180, height=180)

    # Overlay percentage text in the center
    text = alt.Chart(source[source["Topic"] == input_text]).mark_text(
        align='center',
        baseline='middle',
        font="Lato",
        fontSize=24,
        fontWeight='bold',
        color='#4A90E2'
    ).encode(
        text=alt.value(f'{percentage} %')
    ).properties(width=180, height=180)

    return plot + text  # Combine the donut chart and text


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

    #filtered_df = df_reshaped[df_reshaped['DATETIME_DTM'].dt.year == selected_year]
    high_risk_set = []

    if selected_year == 'All':
        for datatuple in keydata:
            if datatuple[4] == 'yes':
                high_risk_set.append(datatuple)
        percentage_result = calculate_percentage(high_risk_set, keydata)    
    else:
        for datatuple in fullyeardata:
            if datatuple[4] == 'yes':
                high_risk_set.append(datatuple)
        percentage_result = calculate_percentage(high_risk_set, fullyeardata)        
                  

    st.markdown('High risk instance')


    # Create and display the donut chart
    donut_chart = make_donut(percentage_result, 'Selected Year Data')
    st.altair_chart(donut_chart, use_container_width=True)
    


with col[1]:
    cloud_text = ''
        

    if selected_year == 'All':
        
        year_count_dict = {}
        for year in year_list:
            if year != 'All':
                year_count_dict[year] = 0

        for mon, yr, key, com, hv in keydata:
            year_count_dict[yr] += 1
            if hv == 'yes':
                cloud_text = cloud_text + key + ', '
        st.markdown('Wordcloud of High Risk Situations')
        cloud_text_final = cloud_text[0:(len(cloud_text) - 2)] # get rid of last space and comma
        wordcloud = WordCloud().generate(cloud_text_final)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

        chart_data = pd.DataFrame(
            {
                "Year": year_count_dict.keys(),
                "Instances": year_count_dict.values()
            }
        )

        st.markdown('Number of Total High Value Instances')
        st.bar_chart(chart_data, x="Year", y="Instances")
    else:
        
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_count = [0,0,0,0,0,0,0,0,0,0,0,0]

    #fullyeardata = get_year_data(selected_year, keydata)
        for mon, yr, key, com, hv in fullyeardata:
            if hv == 'yes':
                month_count[int(mon) - 1] = month_count[int(mon) - 1] + 1
                cloud_text = cloud_text + key + ', '
        
        st.markdown('Wordcloud of High Risk Situations in ' + selected_year)
        cloud_text_final = cloud_text[0:(len(cloud_text) - 2)] # get rid of last space and comma
        wordcloud = WordCloud().generate(cloud_text_final)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        
        chart_data = pd.DataFrame(
            {
                "Month": months, 
                "Instances": month_count,
            }
        )
        st.markdown('Number of High Value Instances in ' + selected_year)
        st.bar_chart(chart_data, x="Month", y="Instances")



with col[2]:
    st.markdown('Featured comments from high risk instances: ')
    
    for i in range(3):
        st.markdown(high_risk_set[random.randrange(0,len(high_risk_set) - 1)][3])


st.dataframe(df_reshaped)