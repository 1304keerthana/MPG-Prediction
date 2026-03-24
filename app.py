import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🚗 MPG Prediction (Polynomial Regression)")

file = st.file_uploader("Upload Auto MPG CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # 🔥 CLEAN COLUMN NAMES
    df.columns = df.columns.str.strip().str.lower()

    st.subheader("🧾 Column Names")
    st.write(list(df.columns))

    # 🔥 AUTO DETECT COLUMNS
    col_map = {
        "mpg": None,
        "displacement": None,
        "horsepower": None,
        "weight": None,
        "acceleration": None
    }

    for col in df.columns:
        if "mpg" in col:
            col_map["mpg"] = col
        elif "disp" in col:
            col_map["displacement"] = col
        elif "horse" in col:
            col_map["horsepower"] = col
        elif "weight" in col:
            col_map["weight"] = col
        elif "acc" in col:
            col_map["acceleration"] = col

    st.write("🔍 Detected Columns:", col_map)

    # Check if all found
    if None in col_map.values():
        st.error("❌ Could not detect required columns. Please check dataset.")
    else:
        try:
            features = [
                col_map["displacement"],
                col_map["horsepower"],
                col_map["weight"],
                col_map["acceleration"]
            ]
            target = col_map["mpg"]

            # Convert to numeric
            for col in features:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df[target] = pd.to_numeric(df[target], errors='coerce')

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

            y_pred = model.predict(X_test)

            # Results
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

            st.success("✅ Model built successfully!")

        except Exception as e:
            st.error(f"Error: {e}")
