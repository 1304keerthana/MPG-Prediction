import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🚗 MPG Prediction (Polynomial Regression)")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Clean column names
    df.columns = df.columns.str.strip()

    st.subheader("🧾 Column Names")
    st.write(list(df.columns))

    # 🔥 USER SELECTS COLUMNS (NO ERROR GUARANTEED)
    st.subheader("🔧 Select Columns")

    target = st.selectbox("Select Target (MPG)", df.columns)

    features = st.multiselect(
        "Select Features (choose 4)",
        df.columns
    )

    if len(features) == 4:
        try:
            # Convert to numeric
            for col in features:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df[target] = pd.to_numeric(df[target], errors='coerce')

            df = df[features + [target]].dropna()

            X = df[features]
            y = df[target]

            # Polynomial
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_poly, y, test_size=0.2, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            st.subheader("📊 Performance")
            st.write("MSE:", mean_squared_error(y_test, y_pred))
            st.write("R²:", r2_score(y_test, y_pred))

            # Plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            st.success("✅ Model built successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning("⚠️ Please select exactly 4 feature columns")
