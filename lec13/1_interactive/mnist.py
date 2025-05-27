# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

# TensorFlowのログレベルを設定
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = DEBUG, 1 = INFO, 2 = WARNING, 3 = ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# MNISTデータセットを読み込む
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# データの前処理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# モデルの構築
def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# モデルのコンパイル
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# CPUでのトレーニングと評価
def train_and_evaluate_on_cpu():
    with tf.device('/CPU:0'):
        model = build_model()
        compile_model(model)
        start_time = time.time()
        model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=1)
        cpu_time = time.time() - start_time
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        predictions = model.predict(test_images)
    return cpu_time, test_acc, predictions

# GPUでのトレーニングと評価
def train_and_evaluate_on_gpu():
    with tf.device('/GPU:0'):
        model = build_model()
        compile_model(model)
        start_time = time.time()
        model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=1)
        gpu_time = time.time() - start_time
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        predictions = model.predict(test_images)
    return gpu_time, test_acc, predictions

# 実行環境のチェック
def is_gpu_available():
    return tf.config.list_physical_devices('GPU')

# 正解と不正解の予測を表示する関数
def display_predictions(test_images, test_labels, predictions, num_examples=10):
    correct_indices = []
    correct_predictions = []
    correct_true_labels = []
    
    incorrect_indices = []
    incorrect_predictions = []
    incorrect_true_labels = []
    
    for i in range(len(test_images)):
        true_label = np.argmax(test_labels[i])
        predicted_label = np.argmax(predictions[i])
        if true_label == predicted_label:
            correct_indices.append(i)
            correct_predictions.append(predicted_label)
            correct_true_labels.append(true_label)
        else:
            incorrect_indices.append(i)
            incorrect_predictions.append(predicted_label)
            incorrect_true_labels.append(true_label)
    
    print("\nCorrect Predictions:")
    print("True Labels:    ", correct_true_labels[:num_examples])
    print("Predictions:    ", correct_predictions[:num_examples])
    print("Indices:        ", correct_indices[:num_examples])
    
    print("\nIncorrect Predictions:")
    print("True Labels:    ", incorrect_true_labels[:num_examples])
    print("Predictions:    ", incorrect_predictions[:num_examples])
    print("Indices:        ", incorrect_indices[:num_examples])

# separator
def print_separator(length=24):
    print(f"%{'-' * length}")

if __name__ == "__main__":
    print("Running on CPU...")
    cpu_time, cpu_acc, cpu_predictions = train_and_evaluate_on_cpu()
    print(f"CPU Training time: {cpu_time:.2f} seconds")
    print(f"CPU Test accuracy: {cpu_acc:.4f}")
    
    display_predictions(test_images, test_labels, cpu_predictions)

    print_separator()
    
    if is_gpu_available():
        print("Running on GPU...")
        gpu_time, gpu_acc, gpu_predictions = train_and_evaluate_on_gpu()
        print(f"GPU Training time: {gpu_time:.2f} seconds")
        print(f"GPU Test accuracy: {gpu_acc:.4f}")
        
        display_predictions(test_images, test_labels, gpu_predictions)
    else:
        print("No GPU available.")


