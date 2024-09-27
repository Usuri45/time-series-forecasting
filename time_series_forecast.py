import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Time Series Forecasting App")
st.write("""
Upload a dataset with a time column and a target column (e.g., sales or stock prices), 
select the appropriate columns, and forecast future values.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    time_col = st.selectbox("Select Time Column", df.columns)
    target_col = st.selectbox("Select Target Column", df.columns)

    time_unit = st.selectbox("Select Time Unit", ["Day", "Week", "Month"])
    periods = st.number_input("Number of Periods to Forecast", min_value=1, max_value=365, value=30)

    df = df[[time_col, target_col]].rename(columns={time_col: 'ds', target_col: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])

    if st.button("Run Forecast"):
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq=time_unit[0].lower())
        forecast = model.predict(future)

        st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
