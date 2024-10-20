import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

# Streamlit page configuration
st.set_page_config(
    page_title="AEP Safeties Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable Altair's dark theme
alt.themes.enable("dark")

# Load the dataset and ensure 'DATETIME_DTM' is in datetime format
df_reshaped = pd.read_csv("static/files/CORE_HackOhio_subset_cleaned_downsampled_1.csv")
df_reshaped['DATETIME_DTM'] = pd.to_datetime(df_reshaped['DATETIME_DTM'], errors='coerce')
df_reshaped = df_reshaped.dropna(subset=['DATETIME_DTM'])  # Drop invalid dates

comments = df_reshaped['PNT_ATRISKNOTES_TX'].tolist()

# Sidebar configuration
with st.sidebar:
    st.title("Types of Load")

    # Extract and sort unique years
    year_list = sorted(df_reshaped['DATETIME_DTM'].dt.year.unique())
    category_list = ['HSIF', 'Capacity', 'PSIF', 'Success', 'Exposure', 'LSIF', 'Low severity']

    # Create selectboxes for year and category selection
    selected_year = st.selectbox('Select a year', year_list, index=0)
    selected_category = st.selectbox('Select a category', category_list, index=0)

# Filter data based on the selected year
filtered_df = df_reshaped[df_reshaped['DATETIME_DTM'].dt.year == selected_year]

def calculate_high_level_incidents():
    # Placeholder model: Calculate high-level incidents
    return len(comments) - 55

# Function to calculate the percentage
def calculate_percentage(df_all, df_selected):
    if len(df_all) == 0:  # Handle empty dataset
        return 0.0
    percentage = (len(df_selected) / len(df_all)) * 100  # Calculate percentage
    return round(percentage, 2)  # Round to two decimals

# Function to create an enhanced donut chart
def make_donut(percentage, input_text):
    # Define colors for chart
    chart_colors = ['#4A90E2', '#E0E0E0']  # Blue and light gray for contrast

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

# Layout with three columns
col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')

# Content in the first column
with col1:
    st.markdown('## Number of Instances')

    # Calculate high-level and low-level incidents
    hl_incidents = calculate_high_level_incidents()
    ll_incidents = len(comments) - hl_incidents

    # Display metrics
    st.metric(label="High Risk Instances", value=str(hl_incidents), delta="5")
    st.metric(label="Low Risk Instances", value=str(ll_incidents), delta="-5")

    st.markdown('High risk instance')

    # Calculate percentage for the selected year
    percentage_result = calculate_percentage(df_reshaped, filtered_df)

    # Create and display the donut chart
    donut_chart = make_donut(percentage_result, 'Selected Year Data')
    st.altair_chart(donut_chart, use_container_width=True)

# Content in the second column
with col2:
    # Prepare data for bar chart
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_count = [0] * 12  # Initialize count for each month

    # Filter data by year and count occurrences per month
    year_data = df_reshaped[df_reshaped['DATETIME_DTM'].dt.year == selected_year]
    for date in year_data['DATETIME_DTM']:
        month_count[date.month - 1] += 1

    # Create DataFrame for the bar chart
    chart_data = pd.DataFrame({
        "Month": months,
        "Incidents": month_count
    })

    # Display bar chart
    st.bar_chart(chart_data, x="Month", y="Incidents")
