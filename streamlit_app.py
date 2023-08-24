import streamlit as st
import joblib
import re
import pandas as pd

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

# Main function to create the Streamlit app
def main():
    st.title("Coffee Quality Prediction App")

    # Sidebar to select the number of features
    num_features = st.sidebar.radio("Select Number of Features:", [2, 3])

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

    selected_model_file = st.selectbox("Select Model:", model_files)

    # Load the selected model
    selected_model = load_model(selected_model_file)

    # Extract feature names from the selected model filename
    feature_names = extract_feature_names(selected_model_file)

    st.write("You have selected:", selected_model_file)
    st.write("Features:", feature_names)

    # Input fields for user interaction
    st.header("Enter Model Details")

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

if __name__ == "__main__":
    main()
