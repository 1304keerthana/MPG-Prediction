import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🚗 MPG Prediction (Polynomial Regression)")

# Upload file
file = st.file_uploader("Upload Auto MPG CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # 🔥 CLEAN COLUMN NAMES (VERY IMPORTANT)
    df.columns = df.columns.str.strip().str.lower()

    # Show columns
    st.subheader("🧾 Column Names")
    st.write(list(df.columns))

    try:
        # Correct column names for most datasets
        features = ['displacement', 'horsepower', 'weight', 'acceleration']
        target = 'mpg'

        # Convert horsepower to numeric (some datasets have '?')
        df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

        df = df[features + [target]].dropna()

        X = df[features]
        y = df[target]

        # Polynomial transformation
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluation
        st.subheader("📊 Model Performance")
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

        # Plot
        st.subheader("📉 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual MPG")
        ax.set_ylabel("Predicted MPG")
        st.pyplot(fig)

        st.success("✅ Model built successfully using Polynomial Regression!")

    except KeyError:
        st.error("❌ Column mismatch. Please check column names above.")
