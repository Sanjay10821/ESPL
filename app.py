import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load dataset
file_path = r"D:\\Naan Mudhalvan\\House Price India.csv"
df = pd.read_csv(file_path)

# Fill NA only for numeric columns
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

# Optional: Remove extreme outliers in Price
price_cap = df['Price'].quantile(0.95)
df = df[df['Price'] <= price_cap]

# Feature and target variables
X_raw = df.drop(columns=['id', 'Price', 'Date', 'Postal Code', 'Renovation Year'])
y = df['Price']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define model evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return model, mae, mse, rmse, r2, y_pred, y_test

# Initialize models
models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=0.01),
    RandomForestRegressor(n_estimators=100, random_state=42),
    GradientBoostingRegressor(n_estimators=100, random_state=42),
    xgb.XGBRegressor(n_estimators=100, random_state=42)
]

# Evaluate models and collect results
results = []
for model in models:
    model, mae, mse, rmse, r2, y_pred, y_test_eval = evaluate_model(model, X_train, X_test, y_train, y_test)
    results.append({
        'Model': model.__class__.__name__,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Predictions': y_pred
    })

# Create DataFrame for evaluation metrics
results_df = pd.DataFrame(results)

# Streamlit App
if __name__ == "__main__":
    def streamlit_dashboard():
        st.title("🏠 House Price Prediction")

        model_choice = st.sidebar.selectbox("Choose a model", [m.__class__.__name__ for m in models])
        model_template = next(m for m in models if m.__class__.__name__ == model_choice)
        selected_model = model_template.fit(X_train, y_train)

        area = st.slider("Living Area (sq ft)", 500, 10000, 1500)
        bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)

        # Create default input dictionary
        input_dict = dict.fromkeys(X_raw.columns, 0)

        # Assign actual user input
        if 'living area' in X_raw.columns:
            input_dict['living area'] = area
        if 'number of bedrooms' in X_raw.columns:
            input_dict['number of bedrooms'] = bedrooms
        if 'number of bathrooms' in X_raw.columns:
            input_dict['number of bathrooms'] = bathrooms

        # Assign sensible defaults to other important features
        defaults = {
            'grade of the house': 8,
            'condition of the house': 5,
            'number of floors': 2,
            'Area of the basement': 1000,
            'lot area': 4000,
            'Built Year': 2000,
            'Number of schools nearby': 3,
            'Distance from the airport': 30,
            'Lattitude': 52.88,
            'Longitude': -114.47
        }

        for key, val in defaults.items():
            if key in input_dict:
                input_dict[key] = val


        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        prediction = selected_model.predict(input_scaled)
        st.success(f"Predicted Price: ₹{prediction[0]:,.2f}")

        st.subheader("📊 Model Evaluation & Plots")
        tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Residual Plot", "Feature Importance"])

        _, _, _, _, _, y_pred_plot, y_test_plot = evaluate_model(selected_model, X_train, X_test, y_train, y_test)

        with tab1:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.scatter(y_test_plot, y_pred_plot, alpha=0.5)
            ax1.plot([y_test_plot.min(), y_test_plot.max()], [y_test_plot.min(), y_test_plot.max()], 'r--')
            ax1.set_xlabel("Actual Price")
            ax1.set_ylabel("Predicted Price")
            ax1.set_title("Actual vs Predicted House Prices")
            st.pyplot(fig1)

        with tab2:
            residuals = y_test_plot - y_pred_plot
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=y_pred_plot, y=residuals, ax=ax2)
            ax2.axhline(0, color='red', linestyle='--')
            ax2.set_xlabel("Predicted Price")
            ax2.set_ylabel("Residual")
            ax2.set_title("Residuals vs Predicted Prices")
            st.pyplot(fig2)

        with tab3:
            if hasattr(selected_model, 'feature_importances_'):
                importances = selected_model.feature_importances_
                feature_names = X_raw.columns
                importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                importance_df = importance_df.sort_values(by='Importance', ascending=False)

                fig3, ax3 = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax3)
                ax3.set_title(f"{selected_model.__class__.__name__} - Feature Importance")
                st.pyplot(fig3)
            else:
                st.info("Feature importance is not available for this model.")

    streamlit_dashboard()