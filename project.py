import streamlit as st
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import io
import os

# Paths
MODEL_PATH = "C:/Users/ghaza/Desktop/Discovery Week/Animal-Detector/sound.tflite"
LABELS_PATH = "C:/Users/ghaza/Desktop/Discovery Week/Animal-Detector/labels.txt"
IMAGES_FOLDER = "C:/Users/ghaza/Desktop/Discovery Week/Animal-Detector/images/"

# Load labels
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Streamlit UI
st.title("🎤Animal Sound Detector")
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
    # Read the WAV file
    samplerate, audio = wav.read(io.BytesIO(uploaded_file.read()))
    audio = audio.astype(np.float32)

    # Resize audio to match model input shape
    target_length = input_shape[1] if len(input_shape) > 1 else len(audio)
    audio_resampled = np.resize(audio, (target_length,))
    audio_data = np.expand_dims(audio_resampled, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Sort predictions by confidence (highest first)
    sorted_indices = np.argsort(output_data[0])[::-1]  
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_confidences = output_data[0][sorted_indices]

    # Ignore "Background Noise" and select the highest valid animal class
    detected_animal = None
    confidence = None

    for i, label in enumerate(sorted_labels):
        if "background noise" not in label.lower():  # Ignore background noise
            detected_animal = label
            confidence = sorted_confidences[i]
            break  

    if detected_animal:
        st.success(f"🐾 Detected Animal: {detected_animal} ({confidence:.2%} confidence)")

        # Try both .jpg and .png formats
        jpg_path = os.path.join(IMAGES_FOLDER, f"{detected_animal}.jpg")
        png_path = os.path.join(IMAGES_FOLDER, f"{detected_animal}.png")

        if os.path.exists(jpg_path):
            st.image(jpg_path, caption=f"{detected_animal}", use_container_width=True)

        elif os.path.exists(png_path):
           st.image(png_path, caption=f"{detected_animal}", use_container_width=True)

        else:
            st.warning(f"❌ No image found for: {detected_animal}")
    else:
        st.warning("❌ No valid animal detected.")
# Footer
st.markdown("""
    <div class='footer'>
        Created by ❤️ Ghazaleh & Dasha
    </div>
""", unsafe_allow_html=True)
