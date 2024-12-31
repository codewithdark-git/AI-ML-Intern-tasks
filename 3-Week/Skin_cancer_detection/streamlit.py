import streamlit as st
from pipeline import MelanomaPipeline  # Import your class from the module
import cv2
import numpy as np
from PIL import Image

def main():
    # Streamlit title with a more compelling format
    st.title("🌟 Melanoma Prediction Demo 🌟")

    # Enhanced description with Markdown for better readability
    st.markdown("""
    Welcome to the **Melanoma Detection Demo**! 👩‍⚕️👨‍⚕️

    This model uses a trained deep learning algorithm to detect **melanoma** from skin lesion images. Here's how it works:

    - **Step 1**: Upload an image of a skin lesion.
    - **Step 2**: The model will analyze the image and classify it as either **Benign** or **Melanoma**.
    - **Step 3**: The prediction along with the confidence score will be displayed.

    > **Disclaimer**: This is a demonstration tool, and **NOT** a replacement for medical advice. Please consult a healthcare professional for any medical concerns.
    
    📷 **Upload an image to begin the detection process.**
    """)

    # Disclaimer section
    st.markdown("#### **⚠️ WARNING! DISCLAIMER! ⚠️**")
    st.markdown("This tool is for **demonstration purposes only** and should **not** be used for medical decision-making.")
    st.markdown("Consult with a qualified healthcare provider for any medical concerns regarding melanoma or skin lesions.")

    # Step 1: Upload the image using Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Step 2: Convert the uploaded image into a format that can be processed
        image = Image.open(uploaded_file)

        # Create two columns for layout
        col1, col2 = st.columns(2)  # Create two columns

        # Display the image in the first column
        with col1:
            st.image(image, caption="Uploaded Image")

        # Convert the image to a format that OpenCV can handle (required by your model)
        img_array = np.array(image.convert('RGB'))
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale as your model expects

        # Save the image temporarily to a file
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, img)

        # Step 3: Run the model prediction
        demo = MelanomaPipeline()  # Instantiate the class
        prediction, confidence = demo.pipeline(temp_image_path)  # Get the prediction and confidence

        # Step 4: Show the result (prediction and confidence) in Streamlit using st.columns
        with col2:
            if prediction and confidence:
                st.metric(label="Prediction", value=prediction)  # Display prediction in the second column
                st.metric(label="Confidence", value=f"{confidence*100} %")  # Display confidence in the second column
            else:
                st.write("An error occurred. Could not make a prediction.")

if __name__ == "__main__":
    main()
