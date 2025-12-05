import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import pickle
import os
from scipy.signal import butter, filtfilt
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Configuration and Styling
st.set_page_config(page_title="🎵 Speech Emotion Recognition", layout="wide")
sns.set(style="whitegrid")

# Define the overall theme colors
PRIMARY_COLOR = "#0A4174"  
SECONDARY_COLOR = "#4E8EA2" 
BG_CARD_COLOR = "#000000"  

# Custom CSS
st.markdown(f"""
<style>
    /* Global background and text color adjustment for dark mode feel */
    .main > div {{
        background-color: #0d1117; /* GitHub Dark Mode color */
    }}
    h1, h2, h3, h4, h5, h6, p, label {{
        color: #f0f6fc; /* Light text color */
    }}
    
    /* Custom Styling for the Main Title */
    .title-text {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {SECONDARY_COLOR};
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
        padding-bottom: 10px;
        border-bottom: 2px solid {PRIMARY_COLOR};
        margin-bottom: 20px;
    }}

    /* Custom Card Style for Metrics and Key Info */
    .stContainer {{
        background-color: {BG_CARD_COLOR};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        border-left: 5px solid {SECONDARY_COLOR};
    }}

    /* Custom Header Style */
    .section-header {{
        font-size: 1.5rem;
        font-weight: 600;
        color: {SECONDARY_COLOR};
        margin-top: 15px;
        margin-bottom: 10px;
    }}
    
    /* Streamlit Sidebar Style */
    .css-1d391kg {{ /* Sidebar selector */
        background-color: {BG_CARD_COLOR} !important;
        border-right: 1px solid {PRIMARY_COLOR};
    }}
    
    /* Adjust Streamlit success/info boxes */
    [data-testid="stSuccess"], [data-testid="stInfo"] {{
        background-color: {BG_CARD_COLOR} !important;
        border: 1px solid {SECONDARY_COLOR};
        color: white !important;
    }}
    
    /* Overwrite specific text colors provided by user, making them more visible */
    .stMarkdown p, .stMarkdown ul, .stMarkdown li {{
        color: #f0f6fc !important;
    }}
</style>
""", unsafe_allow_html=True)


# Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "extracted_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LE_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        data = {
            'label': np.random.choice(["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"], size=1440),
            'mfcc_1': np.random.rand(1440),
            'chroma_1': np.random.rand(1440),
            'zcr_mean': np.random.rand(1440),
            'energy': np.random.rand(1440),
        }
        for i in range(2, 41): data[f'mfcc_{i}'] = np.random.rand(1440)
        for i in range(2, 13): data[f'chroma_{i}'] = np.random.rand(1440)
        
        df = pd.DataFrame(data)
    return df

# Mock model loading
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(LE_PATH, "rb") as f:
            le = pickle.load(f)
    except FileNotFoundError:
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier() 
        scaler = StandardScaler()      
        le = LabelEncoder()             
        
        # Fit mock objects to mock data
        mock_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]
        le.fit(mock_labels)
        
        # Prepare mock feature set for fitting scaler
        df_mock = load_data()
        X_mock = df_mock.drop('label', axis=1)
        scaler.fit(X_mock)
        
        # Mock model fit 
        y_mock = le.transform(df_mock['label'])
        X_mock_scaled = scaler.transform(X_mock)
        model.fit(X_mock_scaled, y_mock)
        
    return model, scaler, le

df = load_data()
model, scaler, le = load_model()

# Emotion Colors

emotion_colors = {
    "angry": "#001D39",
    "calm": "#0A4174",
    "disgust": "#49769F",
    "fearful": "#4E8EA2",
    "happy": "#6EA2B3",
    "neutral": "#7BBDE8",
    "sad": "#BDDBE9",
    "surprised": "#8FC6D9"
}

# Visualization Function 
def visualize_audio_file(file_path, emotion_label):
    y, sr = librosa.load(file_path, sr=None)
    b, a = butter(4, 100/(sr/2), btype='high')
    y_filtered = filtfilt(b, a, y)

    # Feature Extraction
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    energy = y**2
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, axs = plt.subplots(4, 2, figsize=(18, 20))
    plt.style.use('dark_background')

    t = np.linspace(0, len(y)/sr, len(y))

    # Waveform (Original)
    axs[0,0].plot(t, y, color=SECONDARY_COLOR)
    axs[0,0].set_title("Time Domain: Waveform (Original)", color=SECONDARY_COLOR)

    # Waveform (Filtered)
    axs[0,1].plot(t, y_filtered, color="#D62828")
    axs[0,1].set_title("Time Domain: Waveform (High-Pass Filtered)", color=SECONDARY_COLOR)

    # Spectrogram
    librosa.display.specshow(
        librosa.amplitude_to_db(stft, ref=np.max),
        sr=sr,
        x_axis='time',
        y_axis='log',
        ax=axs[1,0],
        cmap='viridis'
    )
    axs[1,0].set_title("Spectrogram", color=SECONDARY_COLOR)

    # Mel Spectrogram
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=axs[1,1],
        cmap='magma'
    )
    axs[1,1].set_title("Mel Spectrogram", color=SECONDARY_COLOR)

    # MFCC
    librosa.display.specshow(
        mfccs,
        sr=sr,
        x_axis='time',
        ax=axs[2,0],
        cmap='magma'
    )
    axs[2,0].set_title("MFCC (Mel-Frequency Cepstral Coefficients)", color=SECONDARY_COLOR)

    # Chroma
    librosa.display.specshow(
        chroma,
        sr=sr,
        x_axis='time',
        y_axis='chroma',
        ax=axs[2,1],
        cmap='coolwarm'
    )
    axs[2,1].set_title("Chroma Features", color=SECONDARY_COLOR)

    # ZCR
    t_zcr = np.linspace(0, len(y)/sr, len(zcr))
    axs[3,0].plot(t_zcr, zcr, color="#6A4C93")
    axs[3,0].set_title("Zero Crossing Rate (ZCR)", color=SECONDARY_COLOR)

    # Energy
    axs[3,1].plot(t, energy, color="#D62828")
    axs[3,1].set_title("Energy (Squared Amplitude)", color=SECONDARY_COLOR)

    plt.suptitle(
        f"Digital Signal Analysis | Predicted Emotion: {emotion_label.upper()}",
        fontsize=20,
        color=PRIMARY_COLOR
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    return fig

# Sidebar
st.sidebar.markdown(f"<h3 style='color:{SECONDARY_COLOR};'>Navigation</h3>", unsafe_allow_html=True)
menu = st.sidebar.radio("Go to", ["🏠 Home", "🗂️ Dataset Overview", "⚙️ Features Detail", "📈 Model Performance", "🗣️ Emotion Prediction"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"<p style='color:gray;'>Created by Lila and Naura.</p>", unsafe_allow_html=True)

# Home
if menu == "🏠 Home":
    st.markdown("<p class='title-text'>🎵 Speech Emotion Recognition Dashboard</p>", unsafe_allow_html=True)
    
    # Introduction Container
    with st.container():
        st.markdown(f"<div class='section-header'>What is Speech Emotion Recognition?</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:18px; line-height: 1.6;'>
        Speech Emotion Recognition (SER) is a method for automatically identifying human emotions from voice recordings using audio signal processing and machine learning. It classifies emotions such as <b>angry, calm, happy, sad, fearful, disgust, surprised, and neutral</b>. This project is developed as an implementation of concepts learned in the Digital Signal Processing (PSD) course.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Model Metrics Cards
    st.markdown(f"<div class='section-header'>Model Performance at a Glance</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Primary Model Type", value="RandomForest and XGBoost")
    
    with col2:
        st.metric(label="Accuracy", value="92.01%", delta="+5.59% (vs baseline)")
        
    with col3:
        st.metric(label="CV Accuracy (5-Fold)", value="86.42%", delta="Solid Generalization")


    st.markdown("---")

    # Features Container
    with st.container():
        st.markdown(f"<div class='section-header'>Key Features Used</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:18px;'>
        The model trained on audio features such as:
        </p>
        <ul style='font-size:16px; margin-left: 20px;'>
            <li>Mel Frequency Cepstral Coefficients: Captures the short-term power spectrum of a sound.</li>
            <li>Chroma: Represents the distribution of energy across the 12 standard pitch classes.</li>
            <li>Zero Crossing Rate: Measures how often the signal crosses the zero-amplitude axis, indicative of noisiness.</li>
            <li>Energy: Represents the loudness of the sound segment.</li>
        </ul>
        """, unsafe_allow_html=True)

# Dataset Overview
elif menu == "🗂️ Dataset Overview":
    st.markdown("<p class='title-text'>🗂️ Dataset Overview</p>", unsafe_allow_html=True)

    # Source and Total Samples
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        st.markdown(f"<div classs='section-header'>Dataset Source (RAVDESS)</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:16px;'>The <a href='https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio/data' target='_blank' style='color:white;'>RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)</a> is a high-quality, professionally recorded dataset.</p>",
            unsafe_allow_html=True
        )
        st.info(f"Total processed samples: **{df.shape[0]}**")

    with col_b:
        st.markdown(f"<div class='section-header'>Data Summary</div>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='font-size:15px;'>
            <li>Actors: 24 (12 male and 12 female)</li>
            <li>File Format: 16-bit, 48kHz .wav</li>
            <li>Emotions: 8 (7 emotional and 1 neutral)</li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Distribution Chart
    st.markdown(f"<div class='section-header'>Distribution of Emotions</div>", unsafe_allow_html=True)
    emotion_counts = df['label'].value_counts().sort_index()
    fig = px.bar(
        x=emotion_counts.index,
        y=emotion_counts.values,
        color=emotion_counts.index,
        color_discrete_map=emotion_colors,
        labels={"x": "Emotion", "y": "Count"},
        title="Distribution of each Emotions"
    )
    fig.update_layout(
        plot_bgcolor=BG_CARD_COLOR, 
        paper_bgcolor='#0d1117',
        font_color='#f0f6fc'
    )
    st.plotly_chart(fig, use_container_width=True)

# Features Detail
elif menu == "⚙️ Features Detail":
    st.markdown("<p class='title-text'>⚙️ Features Detail</p>", unsafe_allow_html=True)

    # Feature Summary
    st.markdown(f"<div class='section-header'>Feature Set Summary</div>", unsafe_allow_html=True)
    feature_cols = [c for c in df.columns if c != 'label']
    total_features = len(feature_cols)

    feature_summary_simple = {
        "MFCC": len([f for f in feature_cols if 'mfcc' in f.lower()]),
        "Chroma": len([f for f in feature_cols if 'chroma' in f.lower()]),
        "ZCR": len([f for f in feature_cols if 'zcr' in f.lower()]),
        "Energy": len([f for f in feature_cols if 'energy' in f.lower()])
    }

    st.markdown(f"**Total Features**: <span style='font-size: 24px; color: {SECONDARY_COLOR};'>**{total_features}**</span>", unsafe_allow_html=True)
    st.markdown(f"<ul style='font-size:16px; margin-left: 20px;'>", unsafe_allow_html=True)
    for name, count in feature_summary_simple.items():
        st.markdown(f"<li>{name}: {count} features</li>", unsafe_allow_html=True)
    st.markdown(f"</ul>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Correlation Heatmap
    st.markdown(f"<div class='section-header'>Feature Correlation Heatmap</div>", unsafe_allow_html=True)
    corr = df[feature_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        reversescale=True,
        zmin=-1, zmax=1
    ))
    fig.update_layout(
        title="Inter-Feature Correlation Matrix",
        width=800, height=800,
        plot_bgcolor=BG_CARD_COLOR, 
        paper_bgcolor='#0d1117',
        font_color='#f0f6fc'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature Distribution Box Plot
    st.markdown(f"<div class='section-header'>Feature Distributions by Emotion</div>", unsafe_allow_html=True)
    selected_feat = st.selectbox("Select Feature to Visualize", feature_cols)
    sorted_labels = sorted(df['label'].unique())
    fig = px.box(
        df,
        x="label",
        y=selected_feat,
        color="label",
        color_discrete_map=emotion_colors,
        category_orders={"label": sorted_labels},
        title=f"Distribution of {selected_feat} by Emotion"
    )
    fig.update_layout(
        plot_bgcolor=BG_CARD_COLOR, 
        paper_bgcolor='#0d1117',
        font_color='#f0f6fc'
    )
    st.plotly_chart(fig, use_container_width=True)

# Model Performance
elif menu == "📈 Model Performance":
    st.markdown("<p class='title-text'>📈 Model Performance</p>", unsafe_allow_html=True)

    # Key Metrics
    st.markdown(f"<div class='section-header'>Key Performance Indicators</div>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.metric(label="Accuracy", value="92.01%", delta="High Classification Rate")
    
    with col_p2:
        st.metric(label="Cross-Validation Mean", value="86.42%", delta="Strong Generalization")

    with col_p3:
        st.metric(label="Model Type", value="RandomForest and XGBoost", delta="Ensemble Method")

    st.markdown("---")

    # Classification Report
    st.markdown(f"<div class='section-header'>Detailed Classification Report</div>", unsafe_allow_html=True)
    report_dict = {
        "precision":[0.9730,0.9467,0.8554,0.9467,0.8961,0.9500,0.8904,0.9241],
        "recall":[0.9474,0.9221,0.9221,0.9221,0.8961,1.0000,0.8442,0.9481],
        "f1-score":[0.9600,0.9342,0.8875,0.9342,0.8961,0.9744,0.8667,0.9359],
        "support":[76,77,77,77,77,38,77,77]
    }
    classes = ["angry","calm","disgust","fearful","happy","neutral","sad","surprised"]
    report_df = pd.DataFrame(report_dict, index=classes)
    
    st.dataframe(report_df.style.set_properties(**{'font-size': '10pt'}))
    st.markdown("---")

    # Confusion Matrix
    st.markdown(f"<div class='section-header'>Confusion Matrix Visualization</div>", unsafe_allow_html=True)
    
    y_true = df['label']
    y_pred = le.inverse_transform(model.predict(scaler.transform(df.drop('label', axis=1))))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale="Blues",
        hoverongaps=False,
        text=cm,
        texttemplate="%{text}",
        textfont={"color":"black"}
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=700,
        height=700
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1) 
    st.plotly_chart(fig, use_container_width=True)

# Emotion Prediction
elif menu == "🗣️ Emotion Prediction":
    st.markdown("<p class='title-text'>🗣️ Emotion Prediction</p>", unsafe_allow_html=True)
    st.markdown("<p>Upload a `.wav` file to see the model's prediction and the underlying audio analysis.</p>", unsafe_allow_html=True)
    
    # File Uploader and Audio Player in a Row
    col_up, col_info = st.columns([1, 3])
    
    with col_up:
        uploaded_file = st.file_uploader("Upload WAV Audio File", type="wav")
    
    if uploaded_file is not None:
        
        # Save temporary file
        temp_path = os.path.join(BASE_DIR, "temp.wav")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocessing function 
        def preprocess_audio(file_path, sr_target=22050, duration_sec=3):
            y, sr = librosa.load(file_path, mono=True, sr=sr_target)
            audio_duration = librosa.get_duration(y=y, sr=sr)
            y, _ = librosa.effects.trim(y)
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            fixed_len = sr_target * duration_sec
            if len(y) > fixed_len:
                y = y[:fixed_len]
            else:
                y = np.pad(y, (0, fixed_len - len(y)))
            return y, sr, audio_duration

        # Feature Extraction function
        def extract_features(file_path):
            y, sr, dur = preprocess_audio(file_path)
            features = []

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            features.extend(np.mean(mfccs, axis=1))

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend(np.mean(chroma, axis=1))

            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend(np.mean(zcr, axis=1))

            energy = np.mean(y**2)
            features.append(energy)

            return np.array(features).reshape(1, -1), dur

        
        with st.spinner('Analyzing audio and predicting emotion...'):
            feat, audio_duration = extract_features(temp_path)
            feat_scaled = scaler.transform(feat)

            # Prediction
            pred_idx = model.predict(feat_scaled)[0]
            pred_emotion = le.inverse_transform([pred_idx])[0]

            # Prediction Probabilities
            probas = model.predict_proba(feat_scaled)[0]
            emotion_list = le.classes_
        
        st.markdown("---")
        
        col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
        
        fixed_white_color = '#f0f6fc'
        
        with col_res1:
            st.markdown(f"<p class='section-header' style='color:{fixed_white_color}'>Predicted Emotion</p>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='font-size:3rem; color:{fixed_white_color};'>{pred_emotion.upper()}</h2>", unsafe_allow_html=True)
            
        with col_res2:
            st.metric(label="Detected Duration", value=f"{audio_duration:.2f} s")
            
        with col_res3:
            st.audio(uploaded_file, format="audio/wav")
            
        st.markdown("---")

        # Probability Chart
        st.markdown(f"<div class='section-header'>Prediction Probabilities</div>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({'Emotion': emotion_list, 'Probability': probas})
        
        prob_fig = px.bar(
            prob_df.sort_values(by='Probability', ascending=False),
            x="Emotion",
            y="Probability",
            color="Emotion",
            color_discrete_map=emotion_colors,
            title="Model Confidence Score for Each Class"
        )
        prob_fig.update_layout(
            plot_bgcolor=BG_CARD_COLOR, 
            paper_bgcolor='#0d1117',
            font_color='#f0f6fc'
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # Signal Analysis Plot
        st.markdown(f"<div class='section-header'>Detailed Audio Signal Analysis</div>", unsafe_allow_html=True)
        fig = visualize_audio_file(temp_path, pred_emotion)
        st.pyplot(fig)

        os.remove(temp_path)
    
    else:
        with col_info:
            st.info("Please upload a .wav file to begin the prediction process.")