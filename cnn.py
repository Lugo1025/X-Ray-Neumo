import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import plot_model

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ======================================================
# 1. ORGANIZE DATASET
# ======================================================
def organize_dataset():
    base_dir = '/kaggle/working/dataset'
    os.makedirs(base_dir, exist_ok=True)

    class_paths = {
        'Normal': [
            '/kaggle/input/3-kinds-of-pneumonia/Curated X-Ray Dataset/Normal',
        ],
        'Pneumonia Bacterial': [
            '/kaggle/input/3-kinds-of-pneumonia/Curated X-Ray Dataset/Pneumonia-Bacterial',
        ],
        'Pneumonia Viral': [
            '/kaggle/input/3-kinds-of-pneumonia/Curated X-Ray Dataset/Pneumonia-Viral',
        ]
    }

    for class_name, src_paths in class_paths.items():
        dest_dir = os.path.join(base_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        file_counter = 1
        for src_path in src_paths:
            if not os.path.exists(src_path):
                continue
            for file in os.listdir(src_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src_file = os.path.join(src_path, file)
                    unique_name = f"{class_name.replace(' ', '_')}_{file_counter}{os.path.splitext(file)[1]}"
                    dest_file = os.path.join(dest_dir, unique_name)
                    if not os.path.exists(dest_file):
                        os.symlink(src_file, dest_file)
                        file_counter += 1
    return base_dir

base_dir = organize_dataset()

# ======================================================
# 2. DATA GENERATORS
# ======================================================
def create_data_generators(base_dir, img_size=(224, 224), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
        horizontal_flip=True, fill_mode='nearest', validation_split=0.2)

    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        base_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='training', seed=42)

    val_generator = val_datagen.flow_from_directory(
        base_dir, target_size=img_size, batch_size=batch_size,
        class_mode='categorical', subset='validation', shuffle=False)

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    return train_generator, val_generator, dict(enumerate(class_weights))

train_generator, val_generator, class_weights = create_data_generators(base_dir)

# ======================================================
# 3. FUNCTIONAL CNN MODEL
# ======================================================
def create_model(input_shape=(224, 224, 3), num_classes=3):
    inputs = Input(shape=input_shape, name='input_layer')
    x = Conv2D(32, 3, activation='relu', padding='same', name='block1_conv1')(inputs)
    x = BatchNormalization(name='block1_bn1')(x)
    x = Conv2D(32, 3, activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization(name='block1_bn2')(x)
    x = MaxPooling2D(2, name='block1_pool')(x)
    x = Dropout(0.2, name='block1_dropout')(x)

    x = Conv2D(64, 3, activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization(name='block2_bn1')(x)
    x = Conv2D(64, 3, activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization(name='block2_bn2')(x)
    x = MaxPooling2D(2, name='block2_pool')(x)
    x = Dropout(0.3, name='block2_dropout')(x)

    x = Conv2D(128, 3, activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization(name='block3_bn1')(x)
    x = Conv2D(128, 3, activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization(name='block3_bn2')(x)
    x = MaxPooling2D(2, name='block3_pool')(x)
    x = Dropout(0.4, name='block3_dropout')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = BatchNormalization(name='fc1_bn')(x)
    x = Dropout(0.5, name='fc1_dropout')(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    return Model(inputs, outputs, name='pneumonia_classifier')

model = create_model()
model.compile(optimizer=Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy', AUC(name='auc', multi_label=True)])

_ = model.predict(np.random.rand(1, 224, 224, 3), verbose=0)  # build for Grad-CAM

# ======================================================
# 4. TRAINING
# ======================================================
def train_model(model, train_gen, val_gen, class_weights):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
        ModelCheckpoint('best_model.h5', monitor='val_auc', mode='max', save_best_only=True)
    ]
    return model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples // val_gen.batch_size,
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

history = train_model(model, train_generator, val_generator, class_weights)

# ======================================================
# 5. EVALUATION
# ======================================================
def plot_history(history):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.title("AUC")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

plot_history(history)

def evaluate_model(model, val_generator):
    val_generator.reset()
    y_pred = model.predict(val_generator)
    y_true = val_generator.classes
    y_pred_labels = np.argmax(y_pred, axis=1)
    class_names = list(val_generator.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_labels, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.show()

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(8, 6))
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.savefig("roc_curves.png")
    plt.show()

    print("\nPer-Class AUC:")
    for i, cls in enumerate(class_names):
        print(f"{cls}: {roc_auc[i]:.4f}")

evaluate_model(model, val_generator)

# ======================================================
# 6. SAVE
# ======================================================
def save_model(model):
    model.save('medical_image_classifier_3class.h5', include_optimizer=True)
    if os.path.exists('best_model.h5'):
        best_model = tf.keras.models.load_model('best_model.h5',
                                                custom_objects={'Adam': Adam, 'AUC': AUC})
        _ = best_model.predict(np.random.rand(1, 224, 224, 3), verbose=0)
        best_model.save('medical_image_classifier_3class_best.h5', include_optimizer=True)

save_model(model)
