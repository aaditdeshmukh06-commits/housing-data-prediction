import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Melbourne Housing Explorer",
    layout="wide"
)

st.title("🏠 Melbourne Housing Price Prediction System")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("melb_data.csv")

df = load_data()

# ---------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------

@st.cache_resource
def train_models():

    features = [
        "Rooms",
        "Bathroom",
        "Landsize",
        "Lattitude",
        "Longtitude"
    ]

    data = df[features + ["Price"]].dropna()

    X = data[features]
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    dt_model = DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=50,
        random_state=42
    )

    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    return dt_model, rf_model

dt_model, rf_model = train_models()

# ---------------------------------------------------
# SIDEBAR MENU
# ---------------------------------------------------

menu = st.sidebar.selectbox(
    "Select Menu",
    ["Dataset", "Visualization"]
)

# ---------------------------------------------------
# DATASET PAGE
# ---------------------------------------------------

if menu == "Dataset":

    st.header("📊 Housing Dataset")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.write(df.describe())

# ---------------------------------------------------
# VISUALIZATION PAGE
# ---------------------------------------------------

elif menu == "Visualization":

    st.header("📈 House Price Prediction")

    col1, col2 = st.columns(2)

    with col1:

        rooms = st.slider(
            "Number of Rooms",
            1,
            10,
            3
        )

        bathroom = st.slider(
            "Number of Bathrooms",
            1,
            10,
            2
        )

        landsize = st.number_input(
            "Landsize",
            min_value=1.0,
            value=200.0
        )

    with col2:

        latitude = st.number_input(
            "Latitude",
            value=-37.81
        )

        longitude = st.number_input(
            "Longitude",
            value=144.96
        )

    input_data = pd.DataFrame({
        "Rooms": [rooms],
        "Bathroom": [bathroom],
        "Landsize": [landsize],
        "Lattitude": [latitude],
        "Longtitude": [longitude]
    })

    dt_price = dt_model.predict(input_data)[0]
    rf_price = rf_model.predict(input_data)[0]

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Decision Tree Prediction",
            f"${dt_price:,.0f}"
        )

    with c2:
        st.metric(
            "Random Forest Prediction",
            f"${rf_price:,.0f}"
        )

    st.divider()

    st.subheader("Prediction Comparison")

    fig1, ax1 = plt.subplots(figsize=(8, 4))

    models = ["Decision Tree", "Random Forest"]
    prices = [dt_price, rf_price]

    bars = ax1.bar(models, prices)

    ax1.set_title("Predicted House Price")
    ax1.set_ylabel("Price ($)")

    for bar in bars:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2,
            yval,
            f"${yval:,.0f}",
            ha='center'
        )

    st.pyplot(fig1)

    st.divider()

    st.subheader("Housing Data Visualization")

    graph_option = st.selectbox(
        "Choose Graph",
        [
            "Rooms vs Price",
            "Landsize vs Price"
        ]
    )

    if graph_option == "Rooms vs Price":

        sample = df[["Rooms", "Price"]].dropna()

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.scatter(
            sample["Rooms"],
            sample["Price"],
            alpha=0.5
        )

        ax2.set_title("Rooms vs House Price")
        ax2.set_xlabel("Rooms")
        ax2.set_ylabel("Price ($)")
        ax2.grid(True)

        st.pyplot(fig2)

    elif graph_option == "Landsize vs Price":

        sample = df[["Landsize", "Price"]].dropna()

        fig3, ax3 = plt.subplots(figsize=(10, 5))

        ax3.scatter(
            sample["Landsize"],
            sample["Price"],
            alpha=0.5
        )

        ax3.set_title("Landsize vs House Price")
        ax3.set_xlabel("Landsize")
        ax3.set_ylabel("Price ($)")
        ax3.grid(True)

        st.pyplot(fig3)