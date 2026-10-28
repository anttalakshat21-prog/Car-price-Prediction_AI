"""
AI Application Lab — Car Price Prediction (Section 216B)
Run:
    python -m pip install streamlit pandas numpy scikit-learn matplotlib
    python -m streamlit run app.py

Upload a used-car CSV containing a price/target column and features such as
brand, mileage, engine type, and vehicle age. This app trains Linear Regression
or Random Forest Regressor and reports MAE and RMSE on a held-out test set.
"""

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AI Application Lab | Car Price Prediction",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp {
  background: linear-gradient(135deg, #f7f8ff 0%, #f3f5fb 55%, #eef4ff 100%);
}
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1450px; }
.hero {
  padding: 2rem 2.2rem; border-radius: 22px; color: white;
  background: linear-gradient(115deg, #172554, #4338ca 65%, #7c3aed);
  box-shadow: 0 14px 35px rgba(49,46,129,.18); margin-bottom: 1.25rem;
}
.hero h1 { color: white; font-size: 2.35rem; margin: .2rem 0 .55rem 0; }
.hero p { color: #e0e7ff; font-size: 1rem; margin-bottom: .2rem; }
.eyebrow {
  text-transform: uppercase; letter-spacing: .14em; font-size: .75rem;
  font-weight: 800; color: #c4b5fd;
}
div[data-testid="stMetric"] {
  background: white; border: 1px solid #e4e8f4; border-radius: 16px;
  padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(30,41,59,.04);
}
.panel {
  background: white; border: 1px solid #e4e8f4; border-radius: 18px;
  padding: 1.2rem 1.3rem; margin-bottom: 1rem;
}
.small-muted { color: #64748b; font-size: .9rem; }
</style>
<div class="hero">
  <div class="eyebrow">AI APPLICATION LAB · SECTION 216B</div>
  <h1>🚘 Car Price Prediction</h1>
  <p>Explore used-car data, train a regression model, and estimate a fair resale price.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧪 Lab controls")
    st.caption("Project 32 · Regression")
    uploaded = st.file_uploader("Upload used-car dataset (.csv)", type=["csv"])
    st.markdown("---")
    st.markdown("### Project objective")
    st.write("Predict the resale price of a used car from details such as brand, mileage, engine type, and vehicle age.")
    st.markdown("### Models")
    model_name = st.selectbox("Choose ML technique", ["Random Forest Regressor", "Linear Regression"])
    test_size = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5)
    st.caption("MAE and RMSE are calculated on the held-out test set.")

@st.cache_data(show_spinner=False)
def read_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

def clean_column_names(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame

def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def money(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.2f}"

if uploaded is None:
    st.info("👈 Upload a CSV file in the sidebar to begin. If you do not have a dataset yet, you can use a Kaggle used-car dataset and upload its CSV here.")
    st.markdown("### What this lab does")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="panel"><h3>📂 Upload</h3><p>Read your used-car CSV and inspect its columns, records, and missing values.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="panel"><h3>🤖 Train</h3><p>Train Linear Regression or Random Forest using your selected price target.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="panel"><h3>📈 Evaluate</h3><p>Compare predictions with real prices using MAE and RMSE.</p></div>', unsafe_allow_html=True)
    st.markdown("### Expected columns")
    st.write("Your file should contain a target price column (for example `selling_price`, `price`, or `sellingprice`) and one or more input features. Common features include brand/name, year or age, mileage/kilometers driven, fuel or engine type, transmission, and owner.")
    st.stop()

try:
    df = clean_column_names(read_csv(uploaded.getvalue()))
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("The CSV must contain at least two columns and one data row.")
    st.stop()

# Convert obvious numeric-looking text columns to numbers where possible.
for col in df.columns:
    if df[col].dtype == "object":
        stripped = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        converted = pd.to_numeric(stripped, errors="coerce")
        non_empty = df[col].notna().sum()
        if non_empty and converted.notna().sum() / non_empty >= 0.85:
            df[col] = converted

st.markdown("## 📊 Dataset dashboard")
missing_cells = int(df.isna().sum().sum())
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{len(df):,}")
m2.metric("Columns", f"{len(df.columns):,}")
m3.metric("Missing cells", f"{missing_cells:,}")
m4.metric("Numeric columns", f"{len(df.select_dtypes(include=np.number).columns):,}")

with st.expander("👀 Preview uploaded dataset", expanded=True):
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)
    st.caption(f"Showing up to 100 rows out of {len(df):,}. Uploaded file: {uploaded.name}")

tab_train, tab_predict, tab_quality = st.tabs(["🤖 Train & evaluate", "🚘 Predict a car price", "🧹 Data quality"])

with tab_train:
    st.markdown("### Configure your regression model")
    candidate_targets = [c for c in df.columns if any(word in c.lower() for word in ["price", "selling", "resale", "target", "cost"])]
    default_target = candidate_targets[0] if candidate_targets else df.columns[-1]
    target = st.selectbox("Select the target price column", options=df.columns.tolist(), index=df.columns.get_loc(default_target))
    available_features = [c for c in df.columns if c != target]
    default_features = available_features
    selected_features = st.multiselect("Select input features", available_features, default=default_features)
    st.caption("Tip: include relevant columns such as brand, mileage, engine/fuel type, year or age. Do not include an ID column unless it is meaningful.")

    if not selected_features:
        st.warning("Select at least one input feature.")
        st.stop()

    work = df[selected_features + [target]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])
    # Drop rows with a target value that cannot be used; feature missingness is imputed in the pipeline.
    if len(work) < 5:
        st.error("At least 5 rows with valid numeric target prices are required for a train/test split.")
        st.stop()
    X = work[selected_features]
    y = work[target].astype(float)
    if y.nunique() < 2:
        st.error("The selected target column must contain at least two different price values.")
        st.stop()

    if st.button("🚀 Train model", type="primary", use_container_width=True):
        with st.spinner("Training the regression model..."):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=42
            )
            preprocessor = make_preprocessor(X_train)
            if model_name == "Linear Regression":
                estimator = LinearRegression()
            else:
                estimator = RandomForestRegressor(
                    n_estimators=250, random_state=42, min_samples_leaf=1, n_jobs=-1
                )
            model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            rmse = math.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float("nan")
            st.session_state["car_model"] = model
            st.session_state["car_features"] = selected_features
            st.session_state["car_target"] = target
            st.session_state["car_model_name"] = model_name
            st.session_state["car_metrics"] = (mae, rmse, r2)
            st.session_state["car_test_results"] = pd.DataFrame({
                "Actual price": y_test.to_numpy(),
                "Predicted price": predictions,
                "Absolute error": np.abs(y_test.to_numpy() - predictions),
            }).reset_index(drop=True)
            st.session_state["car_training_rows"] = len(X_train)
            st.session_state["car_test_rows"] = len(X_test)

    if "car_model" in st.session_state and st.session_state.get("car_target") == target and st.session_state.get("car_features") == selected_features and st.session_state.get("car_model_name") == model_name:
        mae, rmse, r2 = st.session_state["car_metrics"]
        st.markdown("### 📈 Model performance")
        a, b, c = st.columns(3)
        a.metric("MAE", money(mae), help="Mean Absolute Error: average absolute difference between actual and predicted prices. Lower is better.")
        b.metric("RMSE", money(rmse), help="Root Mean Squared Error: penalizes larger prediction errors more strongly. Lower is better.")
        c.metric("R² score", f"{r2:.3f}" if np.isfinite(r2) else "—", help="Closer to 1 is generally better; a negative score can occur when predictions are poor.")
        st.caption(f"Model: {model_name} · Training rows: {st.session_state['car_training_rows']:,} · Test rows: {st.session_state['car_test_rows']:,}")
        results = st.session_state["car_test_results"]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Actual vs predicted")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(results["Actual price"], results["Predicted price"], alpha=0.7)
            low = min(results["Actual price"].min(), results["Predicted price"].min())
            high = max(results["Actual price"].max(), results["Predicted price"].max())
            ax.plot([low, high], [low, high], linestyle="--")
            ax.set_xlabel("Actual price")
            ax.set_ylabel("Predicted price")
            ax.set_title("Prediction quality")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("#### Test predictions")
            st.dataframe(results.head(100).style.format({
                "Actual price": "{:,.2f}",
                "Predicted price": "{:,.2f}",
                "Absolute error": "{:,.2f}",
            }), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download test predictions",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="car_price_test_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Choose the target and features, then click **Train model**. Model scores will appear here.")

with tab_predict:
    st.markdown("### Estimate the resale price of a car")
    model = st.session_state.get("car_model")
    if model is None:
        st.info("Train a model in the **Train & evaluate** tab first.")
    else:
        feature_cols = st.session_state["car_features"]
        with st.form("prediction_form"):
            values = {}
            form_cols = st.columns(2)
            for i, col in enumerate(feature_cols):
                series = df[col]
                with form_cols[i % 2]:
                    if pd.api.types.is_numeric_dtype(series):
                        median = float(series.median()) if series.notna().any() else 0.0
                        values[col] = st.number_input(
                            col, value=median, format="%.2f",
                            help=f"Enter the car's {col} value."
                        )
                    else:
                        choices = sorted(series.dropna().astype(str).unique().tolist())
                        if choices:
                            values[col] = st.selectbox(col, choices)
                        else:
                            values[col] = st.text_input(col)
            submitted = st.form_submit_button("💡 Estimate car price", type="primary", use_container_width=True)
        if submitted:
            one_car = pd.DataFrame([values], columns=feature_cols)
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    one_car[col] = pd.to_numeric(one_car[col], errors="coerce")
            prediction = float(model.predict(one_car)[0])
            st.success("Estimated resale price")
            st.markdown(f"<h1 style='color:#4338ca'>₹ {prediction:,.2f}</h1>", unsafe_allow_html=True)
            st.caption("This is a model estimate based on the uploaded data, not a guaranteed market price. Accuracy depends on data quality, sample size, and whether the dataset reflects current market conditions.")

with tab_quality:
    st.markdown("### 🧹 Data quality report")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(df[c].isna().sum()) for c in df.columns],
        "Missing (%)": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    duplicates = int(df.duplicated().sum())
    st.metric("Exact duplicate rows", f"{duplicates:,}")
    if missing_cells:
        st.warning("Missing values were found. The training pipeline imputes missing feature values, but rows without a valid target price are excluded from training.")
    else:
        st.success("No missing cells detected in this uploaded dataset.")

st.markdown("---")
st.caption("AI Application Lab · Project 32: Car Price Prediction · Section 216B · Regression using Linear Regression / Random Forest Regressor")
