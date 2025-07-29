import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from matplotlib import cm

# ============================
# LOAD MODEL (FORCE BUILD)
# ============================
def load_model_with_init(model_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'Adam': tf.keras.optimizers.Adam, 'AUC': tf.keras.metrics.AUC}
    )
    # Force model to build (important for Grad-CAM)
    dummy_input = tf.ones((1, 224, 224, 3))
    _ = model.predict(dummy_input, verbose=0)
    return model

try:
    model = load_model_with_init('medical_image_classifier_3class.h5')
    print("Model successfully loaded and initialized!")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise

class_names = ['Normal', 'Pneumonia Bacterial', 'Pneumonia Viral']

# ============================
# PREPROCESS IMAGE
# ============================
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# ============================
# GRAD-CAM HEATMAP
# ============================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name='block3_conv2'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy()

# ============================
# CREATE HEATMAP OVERLAY
# ============================
def create_heatmap_overlay(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

# ============================
# MAIN PREDICTION FUNCTION
# ============================
def predict_image(img):
    temp_path = "temp_img.png"
    img.save(temp_path)

    try:
        processed = preprocess_image(temp_path)
        preds = model.predict(processed, verbose=0)[0]

        heatmap, pred_index = make_gradcam_heatmap(processed, model)
        heatmap_img = create_heatmap_overlay(temp_path, heatmap)

        heatmap_path = "heatmap.png"
        heatmap_img.save(heatmap_path)

        plt.figure(figsize=(10, 4))
        bars = plt.barh(class_names, preds)
        plt.title('Classification Probabilities', fontsize=12)
        plt.xlabel('Probability')
        plt.xlim(0, 1)
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                     f'{width:.1%}', ha='left', va='center', fontsize=9)
        plt.tight_layout()
        prob_plot_path = 'probabilities.png'
        plt.savefig(prob_plot_path, bbox_inches='tight', dpi=100)
        plt.close()

        return (
            class_names[pred_index], 
            f"{np.max(preds):.1%}",
            prob_plot_path,
            pd.DataFrame({'Class': class_names, 'Probability': preds}),
            heatmap_path
        )
    except Exception as e:
        print(f"Prediction error: {e}")
        return (
            "Error", "0%", None,
            pd.DataFrame({'Class': class_names, 'Probability': [0.33]*3}),
            None
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ============================
# GRADIO INTERFACE
# ============================
with gr.Blocks(title="Pneumonia Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""# Pneumonia Classification from Chest X-rays""")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="X-ray Image", height=300)
            submit_btn = gr.Button("Classify", variant="primary")
        
        with gr.Column():
            diagnosis = gr.Textbox(label="Diagnosis")
            confidence = gr.Textbox(label="Confidence")
            prob_plot = gr.Image(label="Probabilities", height=200)

    with gr.Row():
        with gr.Column():
            bar_plot = gr.BarPlot(
                pd.DataFrame({'Class': class_names, 'Probability': [0.33]*3}),
                x="Class", y="Probability", title="Class Probabilities",
                height=300, width=400
            )
        with gr.Column():
            gradcam_output = gr.Image(label="Grad-CAM Heatmap", height=300)

    submit_btn.click(
        predict_image,
        inputs=image_input,
        outputs=[diagnosis, confidence, prob_plot, bar_plot, gradcam_output]
    )

    gr.Markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px;'>
    <small><b>Disclaimer:</b> For research/educational use only. Not for clinical diagnosis.</small>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=True)
