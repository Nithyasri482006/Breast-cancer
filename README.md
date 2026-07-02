# Breast Cancer Detection Using Deep Learning

## Overview
This project detects breast cancer from ultrasound images using the **MobileNetV2** deep learning model. It classifies images into three categories:
- Normal
- Benign
- Malignant

A **Streamlit** web application is used to upload an image and display the prediction.

## Dataset
- Breast Ultrasound Images Dataset (BUSI)
- Classes: Normal, Benign, Malignant

## Technologies Used
- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- OpenCV
- Pillow
- Streamlit

## Workflow
1. Load the BUSI dataset.
2. Preprocess and resize images (224 × 224).
3. Train the MobileNetV2 model using Transfer Learning.
4. Save the trained model.
5. Deploy the model using Streamlit.

## Project Structure

```
breast_cancer_project/
│── breast_app.py
│── breast_cancer_final_model.keras
│── breast_cancer_v1.ipynb
│── requirements.txt
│── README.md
```

## Run the Project

```bash
pip install -r requirements.txt
streamlit run breast_app.py
```

## Output
The application predicts whether the uploaded breast ultrasound image is:
- Normal
- Benign
- Malignant


# Breast-cancer
