import streamlit as st
import joblib
import re
import pandas as pd

# Intro Page
def intro_page():
    st.title("Coffee Quality Prediction App")
    st.markdown("### Introduction")
    st.write("This app focuses on developiong the ability to use flavor metrics to predict coffee quality to increase the profitablity of crops.")
    st.write("The global coffee industry was valued at 466 Billion dollars in 2020 and is forecasted to continually increase into 2026.")
    st.write("Coffee quality can be determined using a variety of features. This tool uses data from the [Coffee Quality Institute](https://database.coffeeinstitute.org) and machine learning to make predictions of overall quality, deemed 'Total Cup Points', from a reduced amount of features or 'Cupping Scores'.")
    st.image("dot_plot.png", caption="Predicted values are based on 4 features", use_column_width=True)

    st.markdown("### Data")
    st.write("The data that I am using includes more than 1300 occurrences across 8 quality metrics for coffee across continents. The 8 metrics have been reduced to 4 avoiding multicollinearity and overfitting while increasing interpretability and generalization of the model.")
    st.write("- From these 4 numeric metrics, all combinations of 2 and 3 can be used with an associated country name to make reliable predictions on new unseen data with 0.341-0.377 root mean squared error after an 80/20 train test split.")
    st.write("- Country names are one-hot encoded before being introduced to the model. This allows the model to determine how the predicted value will be penalized or boosted based on the presence of a particular country, considering interactions with the numeric features and the overall model structure.")
        
    st.image("intro_image.png", caption="Project Schematic", use_column_width=True)

    st.markdown("### Conclusion")
    st.write("With value of coffee continuing to grow every year, this tool will ensure an increase the competitiveness of American coffee companies globally.")
    
    if st.button("Enter"):
        st.session_state.page = "main_page"

# Main Page
# Function to load a model from a pickle file
def load_model(filename):
    with open(filename, 'rb') as file:
        model = joblib.load(file)
    return model

# Extract feature names from the model filename
def extract_feature_names(filename):
    # Extract the portion of the filename before the ".pkl" extension
    name_without_extension = re.match(r"model_(.+)\.pkl", filename).group(1)
    # Split the name using underscores and remove numeric values
    feature_names = [part for part in name_without_extension.split("_") if not part.isdigit()]
    return feature_names

def main_page():
    st.title("Coffee Quality Prediction App")

    # Add a section with text and bullet points
    st.write("Welcome to the Coffee Predictor App! This app allows you to predict coffee quality based on selected features.")

    st.header("What to do:")
    st.write("- Select the number of features (2 or 3) for prediction.")
    st.write("- Choose a model based on the selected number of features.")
    st.write("- Enter details such as country of origin (below) and feature values (left sidebar).")
    st.write("- Click the 'Make Prediction' button to see the predicted Total Cup Points out of 10")

    st.header("Feature Description")
    st.write("- Aroma = fragrence or smell")
    st.write("- Acidity = brightness or sharpness of flavor adding complexity to the coffee")
    st.write("- Body = weight and texture of the coffee on your tongue and in your mouth")
    st.write("- Balance = harmonious interaction between its various components, including acidity, sweetness, bitterness, and body")
    
    # Sidebar to select the number of features
    num_features = st.sidebar.radio("Select Number of Features:", [2, 3], index=1)

    if num_features == 2:
        model_files = [
            "model_Aroma_Acidity_1.pkl",
            "model_Aroma_Body_2.pkl",
            "model_Aroma_Balance_3.pkl",
            "model_Acidity_Body_4.pkl",
            "model_Acidity_Balance_5.pkl",
            "model_Body_Balance_6.pkl"
        ]
    else:
        model_files = [
            "model_Aroma_Acidity_Body_1.pkl",
            "model_Aroma_Acidity_Balance_2.pkl",
            "model_Aroma_Body_Balance_3.pkl",
            "model_Acidity_Body_Balance_4.pkl"
        ]

    st.header("Model and Feature Selection")
    selected_model_file = st.selectbox("Select Model:", model_files, index=3)

    # Load the selected model
    selected_model = load_model(selected_model_file)

    # Extract feature names from the selected model filename
    feature_names = extract_feature_names(selected_model_file)

    st.write("You have selected:", selected_model_file)
    st.write("Features:", feature_names)

    
    # Example dictionary of model RMSE values
    model_rmse_values = {
        "model_Aroma_Acidity_1.pkl": 0.3647593206059453,
        "model_Aroma_Body_2.pkl": 0.37688258052955415,
        "model_Aroma_Balance_3.pkl": 0.35307895416137697,
        "model_Acidity_Body_4.pkl": 0.35829623252613757,
        "model_Acidity_Balance_5.pkl": 0.34532259446732716,
        "model_Body_Balance_6.pkl": 0.3494482838429561,
        "model_Aroma_Acidity_Body_1.pkl": 0.35900496154653594,
        "model_Aroma_Acidity_Balance_2.pkl": 0.34530955400586516,
        "model_Aroma_Body_Balance_3.pkl": 0.3490474675990926,
        "model_Acidity_Body_Balance_4.pkl": 0.34113254483468036
    
    }

    # Get the RMSE value for the selected model
    selected_model_rmse = model_rmse_values.get(selected_model_file, "RMSE not available")

    st.write("Selected Model RMSE:", selected_model_rmse)

    # Input fields for user interaction
    st.header("Country Selection")

    # Categorical input
    country_of_origin = st.text_input("Country of Origin")

    # Numeric inputs
    feature_values = []
    for feature_name in feature_names:
        # Add sliders for numeric features with specified range
        feature_value = st.sidebar.slider(
            f"{feature_name.capitalize()} Value", 
            min_value=5.0, 
            max_value=10.0, 
            value=5.0,  # Default value
            step=0.1,   # Step size
            key=feature_name
        )
        feature_values.append(feature_value)

    # Make a prediction using the selected model
    if st.button("Make Prediction"):
        data = {
            "CountryOfOrigin": [country_of_origin]
        }

        for feature_name, feature_value in zip(feature_names, feature_values):
            data[feature_name] = [feature_value]

        input_df = pd.DataFrame(data)

        # Make prediction using the selected model
        prediction = selected_model.predict(input_df)

        st.write("Prediction:", prediction)

    if st.button("Go Back to Intro Page"):
        st.session_state.page = "intro_page"

# Main function
def main():
    st.set_page_config(page_title="Streamlit Intro App")
    
    if "page" not in st.session_state:
        st.session_state.page = "intro_page"
    
    if st.session_state.page == "intro_page":
        intro_page()
    elif st.session_state.page == "main_page":
        main_page()

if __name__ == "__main__":
    main()
