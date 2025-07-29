Pneumonia X-ray Classifier with Grad-CAM GUI
============================================

This project provides a Convolutional Neural Network (CNN) for the classification
of chest X-ray images into three categories:

1. Normal
2. Pneumonia Bacterial
3. Pneumonia Viral

Additionally, it includes a Gradio-based GUI to visualize predictions and
Grad-CAM heatmaps highlighting regions of interest in the X-rays.

------------------------------------------------------------
Features
------------------------------------------------------------
- CNN Model (Keras/TensorFlow) for 3-class pneumonia classification.
- Data Augmentation using ImageDataGenerator.
- Performance Metrics: Accuracy, AUC, Confusion Matrix, and ROC Curves.
- Grad-CAM Visualization to interpret CNN decisions.
- Interactive GUI powered by Gradio for easy use:
  * Image upload and classification
  * Probability bar plots
  * Grad-CAM heatmap overlay

------------------------------------------------------------
Project Structure
------------------------------------------------------------
.
├── cnn.py                     # Model training and evaluation
├── gui.py                     # Gradio GUI for prediction and visualization
├── medical_image_classifier_3class.h5  # Trained CNN model
├── requirements.txt           # Python dependencies
├── README.txt                 # Project documentation

------------------------------------------------------------
Installation
------------------------------------------------------------
1. Clone Repository
   git clone https://github.com/yourusername/pneumonia-xray-classifier.git
   cd pneumonia-xray-classifier

2. Create Virtual Environment (Optional but Recommended)
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate

3. Install Requirements
   pip install -r requirements.txt

Key Dependencies:
- Python 3.8+
- TensorFlow 2.10+
- NumPy, Pandas, Matplotlib, Seaborn
- OpenCV
- Gradio

------------------------------------------------------------
Model Training
------------------------------------------------------------
To train the CNN from scratch:

   python cnn.py

The script will:
1. Organize the dataset into Normal, Pneumonia Bacterial, and Pneumonia Viral.
2. Train the model with data augmentation and class balancing.
3. Generate evaluation plots:
   - Accuracy / Loss curves
   - ROC curves
   - Confusion matrix
4. Save the trained model as:
   - medical_image_classifier_3class.h5
   - medical_image_classifier_3class_best.h5

------------------------------------------------------------
GUI Usage
------------------------------------------------------------
Once the model is trained or a pretrained model file is available:

   python gui.py

Upload a chest X-ray image to get:
- Predicted class with confidence
- Probability bar plot
- Grad-CAM heatmap

By default, demo.launch(share=True) allows public sharing of the interface.

------------------------------------------------------------
Example Output
------------------------------------------------------------
1. Probability Bar Plot
   Displays the classification probability for each category.

2. Grad-CAM Heatmap
   Highlights lung regions contributing to the model's decision.

------------------------------------------------------------
Disclaimer
------------------------------------------------------------
For research and educational purposes only.
Not intended for clinical use or medical diagnosis.
