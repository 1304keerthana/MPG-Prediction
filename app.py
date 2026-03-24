import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.title("🚗 Fuel Efficiency Predictor (MPG)")

file = st.file_uploader("Upload Auto MPG CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.write(df.head())

    features = ['Displacement', 'Horsepower', 'Weight', 'Acceleration']
    target = 'MPG'

    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

    # Polynomial transform
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("📊 Model Performance")
    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("R²:", r2_score(y_test, y_pred))

    # Plot
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual MPG")
    plt.ylabel("Predicted MPG")
    st.pyplot(plt)

    st.info("📈 Polynomial regression captures non-linear relationships better!")
