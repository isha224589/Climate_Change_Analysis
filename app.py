import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide", page_title="🌎 Climate Insight by Isha")
st.title("🌎 Climate Insight: An Interactive Dashboard by Isha Singh")
st.markdown("""
Welcome to a personalized climate change dashboard that explores the rise in greenhouse gases and temperature anomalies from 1983–2050. Dive into trends, understand correlations, and forecast the future.
""")

# Load preprocessed data
combined_df = pd.read_csv("climate_change_summary.csv")

# Sidebar controls
st.sidebar.header("📅 Select Year Range")
min_year, max_year = int(combined_df.year.min()), int(combined_df.year.max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))
filtered_df = combined_df[(combined_df["year"] >= year_range[0]) & (combined_df["year"] <= year_range[1])]

# Unique style for dashboard
st.markdown("---")
st.markdown("## 🔍 Climate Trends & Projections")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Greenhouse Gas Trends", "📊 Temperature & Correlation", "🧠 Forecast 2025–2050"])

with tab1:
    st.subheader("📈 Yearly Greenhouse Gas Trends")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### CO₂ Levels")
        fig_co2, ax1 = plt.subplots()
        sns.lineplot(data=filtered_df, x='year', y='co2_avg', ax=ax1, color='forestgreen')
        ax1.set_ylabel("ppm")
        ax1.set_xlabel("Year")
        st.pyplot(fig_co2)

    with col2:
        st.markdown("### CH₄ Levels")
        fig_ch4, ax2 = plt.subplots()
        sns.lineplot(data=filtered_df, x='year', y='ch4_avg', ax=ax2, color='darkorange')
        ax2.set_ylabel("ppb")
        ax2.set_xlabel("Year")
        st.pyplot(fig_ch4)

with tab2:
    st.subheader("🌡️ Temperature Trends and Correlations")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("### Global Temperature Anomaly")
        fig_temp, ax3 = plt.subplots()
        sns.lineplot(data=filtered_df, x='year', y='temp_raw', ax=ax3, color='crimson')
        ax3.set_ylabel("°C")
        ax3.set_xlabel("Year")
        st.pyplot(fig_temp)

    with col4:
        st.markdown("### Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(filtered_df[['co2_avg', 'ch4_avg', 'temp_raw']].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

with tab3:
    st.subheader("🔮 Predict Future Temperature Anomalies")

    model = LinearRegression()
    model.fit(combined_df[["co2_avg", "ch4_avg"]], combined_df["temp_raw"])

    future_years = np.arange(2025, 2051)
    co2_model = LinearRegression().fit(combined_df[["year"]], combined_df[["co2_avg"]])
    ch4_model = LinearRegression().fit(combined_df[["year"]], combined_df[["ch4_avg"]])

    future_df = pd.DataFrame({"year": future_years})
    future_df["co2_avg"] = co2_model.predict(future_df[["year"]].values)
    future_df["ch4_avg"] = ch4_model.predict(future_df[["year"]].values)
    future_df["predicted_temp"] = model.predict(future_df[["co2_avg", "ch4_avg"]])

    st.dataframe(future_df, use_container_width=True)

    st.markdown("### 📈 Temperature Forecast")
    fig_pred, ax_pred = plt.subplots()
    sns.lineplot(data=combined_df, x="year", y="temp_raw", label="Actual", ax=ax_pred)
    sns.lineplot(data=future_df, x="year", y="predicted_temp", label="Predicted", ax=ax_pred, color='darkred')
    ax_pred.set_ylabel("Temperature Anomaly (°C)")
    st.pyplot(fig_pred)

st.markdown("---")
st.caption("Created by Isha Kumari | Data Source: NASA Global Climate Portal")