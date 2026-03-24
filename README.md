# 🏥 PneumoDetect AI - Pediatric Pneumonia Detection

An AI-powered system designed to detect pneumonia from pediatric chest X-ray images (ages 1-5) using deep learning.

### 🌟 **Key Features**
- **High Accuracy**: 86% cross-operator validation accuracy and 96.4% sensitivity.
- **Fast Analysis**: Get results and a professional medical report in under 3 seconds.
- **Medical Grade**: Supports professional DICOM files and standard image formats (JPG, PNG).
- **Explainable AI**: Includes attention maps to show which parts of the X-ray the AI focused on.
- **Dual Interface**: Professional Streamlit dashboard for users and a FastAPI backend for developers.

### 🏗️ **Project Structure**
- **`api/`**: Contains the Streamlit frontend and FastAPI backend code.
- **`scripts/`**: Training, evaluation, and data balancing scripts.
- **`results/`**: Performance charts, confusion matrices, and validation reports.
- **`models/`**: (Generated after training) Stores the trained AI model weights.

### ⚙️ **Brief Functionality**
- **What it does**: The system takes an X-ray image (pediatric) and predicts whether it shows signs of pneumonia or is normal.
- **How it works**: It uses a **MobileNetV2** deep learning model. The image is processed through several layers that identify lung patterns and output a probability score from 0 (Normal) to 1 (Pneumonia).

### 📊 **Data Normalization & Processing**
To ensure the AI works accurately, we normalize the data in both training and testing phases:
- **Normalization**: Every image's pixel values (0-255) are divided by **255.0**. This scales the data to a **0-1 range**, which helps the AI learn faster and maintain mathematical stability.
- **Training Data**: 
    - **Augmentation**: We apply random rotations, zooms, and flips. This teaches the AI to recognize pneumonia regardless of how the X-ray was taken.
- **Test & Live Data**:
    - **Consistency**: Images uploaded via the UI are rescaled to the same **0-1 range** and resized to **224x224 pixels** to perfectly match what the AI learned during its "study" phase.

### 🚀 **Quick Start**
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Model** (Downloads data & trains automatically):
   ```bash
   cd scripts
   python train_full_pipeline.py
   ```
3. **Run the Dashboard**:
   ```bash
   streamlit run api/streamlit_api_folder/streamlit_app.py
   ```
4. **Run the API**:
   ```bash
   uvicorn api.main:app --reload
   ```

### 📊 **Performance Summary**
- **Sensitivity**: 96.4% (Catches 96 out of 100 cases)
- **Specificity**: 74.8% (Correctly identifies healthy lungs)
- **ROC-AUC**: 0.964 (Outstanding diagnostic performance)

---
*Note: This is a research prototype for preliminary screening and is not a replacement for professional medical diagnosis.*
