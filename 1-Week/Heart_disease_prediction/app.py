from urllib import response
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from g4f.client import Client
from sklearn.preprocessing import StandardScaler

# Load the pre-trained models
try:
    with open('Models/DecisionTreeClassifier_model.pkl', 'rb') as f:
        dTCM = pickle.load(f)
    with open('Models/RandomForestClassifier_model.pkl', 'rb') as f:
        rFCM = pickle.load(f)
    with open('Models/LogisticRegression_model.pkl', 'rb') as f:
        lGRM = pickle.load(f)
    with open('Models/SVC_model.pkl', 'rb') as f:
        sVMM = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")

# Initialize scaler
scaler = StandardScaler()

def format_predictions(predictions):
    """Format predictions into a detailed and readable string using GPT-based formatting."""
    # Construct a prompt to guide the GPT-based API
    prompt = """
    Format the following predictions into a detailed, user-friendly report. 
    Each model's prediction should include the model name, prediction (Positive or Negative), and a brief explanation:
    - Positive (1): Indicates the presence of the condition (heart disease).
    - Negative (0): Indicates the absence of the condition.

    Example format:
    - Model Name: Prediction (Positive/Negative). Explanation.

    Predictions: {predictions}
    """.strip()

    client = Client()

    # Generate response from GPT model
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Update model to your preferred GPT-based model
            messages=[{"role": "user", "content": prompt.format(predictions=predictions)}]
        )
        return response.choices[0].message.content  # Return the formatted result
    except Exception as e:
        return f"Error formatting predictions: {str(e)}"


def predict_target(data, best_model_only=False):
    """Make predictions using the loaded models."""
    try:
        X_numpy = np.array(data).reshape(1, -1)
        X_scaled = scaler.fit_transform(X_numpy)
        
        predictions = {
            'Decision Tree': dTCM.predict(X_scaled)[0],
            'Random Forest': rFCM.predict(X_scaled)[0],
            'Logistic Regression': lGRM.predict(X_scaled)[0],
            'Support Vector Machine': sVMM.predict(X_scaled)[0]
        }
        
        if best_model_only:
            return {'Random Forest (Best Model)': predictions['Random Forest']}
        
        return predictions
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Streamlit UI
def app():
    st.title("Heart Disease Prediction App")

    # Display input method options
    input_method = st.selectbox(
        "How would you like to input the data?",
        ["Enter Data Manually", "Upload CSV File"]
    )

    if input_method == "Upload CSV File":
        # CSV file upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.write("CSV File Preview:", df.head())
                
                predictions_list = []
                
                for _, row in df.iterrows():
                    pred = predict_target(row.values.tolist())
                    predictions_list.append(pred)
                
                # Format and display results
                result_message = "Predictions for uploaded data:\n\n"
                for i, pred in enumerate(predictions_list, 1):
                    result_message += f"Patient {i}:\n{format_predictions(pred)}\n"
                
                st.text(result_message)
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")
                
    else:
        # Form for manual data input
        st.header("Enter Patient Data")
        
        # Create a DataFrame for the input fields
        data = pd.DataFrame({
            "age": [0],
            "sex": [0],
            "cp": [0],
            "trestbps": [0],
            "chol": [0],
            "fbs": [0],
            "restecg": [0],
            "thalach": [0],
            "exang": [0],
            "oldpeak": [0.0],
            "slope": [0],
            "ca": [0],
            "thal": [0]
        })

        # Use the data editor for input
        edited_data = st.data_editor(data)

        # Prediction type selection
        pred_type = st.selectbox(
            "Choose prediction type:",
            ["Check All Models", "Use Best Model Only"]
        )

        if st.button("Predict"):
            # Ensure that the data from the editor is used for prediction
            data_values = edited_data.iloc[0].values.tolist()

            # Make prediction
            predictions = predict_target(data_values, best_model_only=(pred_type == "Use Best Model Only"))
            
            # Format and display results
            result_message = format_predictions(predictions)
            st.markdown(result_message)

if __name__ == "__main__":
    app()
