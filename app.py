import streamlit as st
import tensorflow as tf
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import os
from tempfile import NamedTemporaryFile

# Load the trained model
model = tf.keras.models.load_model("lstm_model.h5")

# Define label encoder classes
label_encoder_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

# Function to preprocess the audio for the LSTM model
def preprocess_audio(file_path, max_pad_length=200):
    """
    Preprocess the audio file: load, extract MFCC, and pad.
    """
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.effects.preemphasis(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    padded_mfcc = tf.keras.preprocessing.sequence.pad_sequences([mfcc.T], maxlen=max_pad_length, padding="post")
    return padded_mfcc

# Function to save the recorded audio
def save_audio_file(audio_data, sample_rate, file_path):
    """
    Save the recorded audio as a .wav file.
    """
    sf.write(file_path, audio_data, sample_rate)

# Function to delete temporary audio files
def delete_temp_file(file_path):
    """
    Delete the temporary file if it exists.
    """
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

# Initialize session state for separate audio management
if "recorded_audio_file" not in st.session_state:
    st.session_state.recorded_audio_file = None
if "uploaded_audio_file" not in st.session_state:
    st.session_state.uploaded_audio_file = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

# Streamlit App Layout
st.title("Decoding Emotions From Speech")
st.write("Record your voice or upload an audio file to predict the emotion!")

# --- LAYOUT WITH CARDS ---
st.markdown("### Choose an option below:")
col1, col2 = st.columns(2)

# --- RECORDING BOX ---
with col1:
    st.markdown(
        """
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h3 style="text-align: center; color: #FFFFFF;">Record Audio 🎙️</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Start/Stop Buttons for Recording
    if st.button("Start Recording", key="record_start"):
        delete_temp_file(st.session_state.recorded_audio_file)
        st.session_state.recorded_audio_file = None
        st.session_state.is_recording = True
        duration = 10
        st.session_state.audio_data = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype="float32")
        st.write("Recording... Speak now!")

    if st.session_state.is_recording and st.button("Stop Recording", key="record_stop"):
        sd.stop()
        st.session_state.is_recording = False
        st.write("Recording stopped!")
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            save_audio_file(st.session_state.audio_data, 16000, temp_file.name)
            st.session_state.recorded_audio_file = temp_file.name
        st.audio(st.session_state.recorded_audio_file, format="audio/wav")

    # Predict Button for Recording
    if st.session_state.recorded_audio_file and st.button("Predict Emotion", key="record_predict"):
        try:
            input_features = preprocess_audio(st.session_state.recorded_audio_file)
            predictions = model.predict(input_features)
            predicted_label = label_encoder_classes[np.argmax(predictions)]
            st.success(f"Predicted Emotion: **{predicted_label.capitalize()}** 🎉")
            delete_temp_file(st.session_state.recorded_audio_file)
            st.session_state.recorded_audio_file = None
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- UPLOADING BOX ---
with col2:
    st.markdown(
        """
        <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h3 style="text-align: center; color: #FFFFFF;">Upload Audio 📂</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File Uploader
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        delete_temp_file(st.session_state.uploaded_audio_file)
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state.uploaded_audio_file = temp_file.name
        st.audio(st.session_state.uploaded_audio_file, format="audio/wav")

    # Predict Button for Uploading
    if st.session_state.uploaded_audio_file and st.button("Predict Emotion", key="upload_predict"):
        try:
            input_features = preprocess_audio(st.session_state.uploaded_audio_file)
            predictions = model.predict(input_features)
            predicted_label = label_encoder_classes[np.argmax(predictions)]
            st.success(f"Predicted Emotion: **{predicted_label.capitalize()}** 🎉")
            delete_temp_file(st.session_state.uploaded_audio_file)
            st.session_state.uploaded_audio_file = None
        except Exception as e:
            st.error(f"An error occurred: {e}")
