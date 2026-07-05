"""
Parkinson's Disease Detection System - Frontend
A Streamlit-based GUI for detecting Parkinson's disease through drawing analysis
"""

import streamlit as st
import cv2
import numpy as np
import os
import pickle
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys

# Import the backend system
# Note: Make sure parkinsons_enhanced.py is in the same directory
try:
    from parkinsons_enhanced import ParkinsonsDetectionSystem, create_sample_dirs
except ImportError:
    st.error("⚠️ Backend file 'parkinsons_enhanced.py' not found! Please ensure it's in the same directory.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Parkinson's Disease Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = ParkinsonsDetectionSystem()
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

def add_log(message, level="info"):
    """Add a log message with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append({
        "time": timestamp,
        "level": level,
        "message": message
    })

def display_logs():
    """Display recent logs in the sidebar"""
    if st.session_state.logs:
        st.sidebar.markdown("### 📋 Activity Log")
        for log in st.session_state.logs[-10:]:  # Show last 10 logs
            icon = "ℹ️" if log["level"] == "info" else ("✅" if log["level"] == "success" else "❌")
            st.sidebar.text(f"{icon} [{log['time']}] {log['message']}")

def check_models_exist(dataset_name, model_name):
    """Check if trained models exist for given dataset and model"""
    model_path = f'./models/{dataset_name}/{model_name}_model.pkl'
    scaler_path = f'./models/{dataset_name}/{dataset_name}_scaler.pkl'
    encoder_path = f'./models/{dataset_name}/{dataset_name}_encoder.pkl'
    return all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path])

def get_available_models():
    """Get list of available trained models"""
    available = {'spiral': [], 'wave': []}
    for dataset in ['spiral', 'wave']:
        for model in ['RF', 'SVM', 'KNN']:
            if check_models_exist(dataset, model):
                available[dataset].append(model)
    return available

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("# 🧠 Navigation")
    page = st.radio(
        "Select Page",
        ["🏠 Home", "📊 Predict", "🎯 Train Models", "📈 Compare Results", "🔬 Visualization", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    available_models = get_available_models()
    
    col1, col2 = st.columns(2)
    with col1:
        spiral_count = len(available_models['spiral'])
        st.metric("Spiral Models", spiral_count, delta="✓" if spiral_count > 0 else "✗")
    with col2:
        wave_count = len(available_models['wave'])
        st.metric("Wave Models", wave_count, delta="✓" if wave_count > 0 else "✗")
    
    st.markdown("---")
    display_logs()

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<div class="main-header">🧠 Parkinson\'s Disease Detection System</div>', unsafe_allow_html=True)
    
    # Introduction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3 style="text-align: center;">Welcome to the AI-Powered Parkinson's Detection System</h3>
        <p style="text-align: center;">
        This system uses advanced machine learning algorithms to analyze spiral and wave drawings 
        for early detection of Parkinson's disease. Upload a drawing, and our trained models will 
        provide instant analysis.
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("##")
    
    # Features
    st.markdown('<div class="sub-header">✨ Key Features</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Multiple Datasets
        - **Spiral Drawings**: Analyze spiral patterns
        - **Wave Drawings**: Analyze wave patterns
        - Separate training for each type
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 Three ML Models
        - **Random Forest**: Ensemble learning
        - **SVM**: Support Vector Machine
        - **KNN**: K-Nearest Neighbors
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Advanced Analysis
        - HOG feature extraction
        - Image preprocessing
        - Confidence scores
        - Visual comparisons
        """)
    
    st.markdown("##")
    
    # How it works
    st.markdown('<div class="sub-header">🔍 How It Works</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        #### 1️⃣ Upload
        Upload a spiral or wave drawing image
        """)
    
    with col2:
        st.markdown("""
        #### 2️⃣ Process
        Image is preprocessed and features are extracted
        """)
    
    with col3:
        st.markdown("""
        #### 3️⃣ Analyze
        ML model analyzes the patterns
        """)
    
    with col4:
        st.markdown("""
        #### 4️⃣ Results
        Get prediction with confidence score
        """)
    
    st.markdown("##")
    
    # Quick stats
    st.markdown('<div class="sub-header">📈 Quick Stats</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_models = len(available_models['spiral']) + len(available_models['wave'])
        st.metric("Trained Models", total_models, "out of 6 possible")
    
    with col2:
        prediction_count = len(st.session_state.prediction_history)
        st.metric("Predictions Made", prediction_count)
    
    with col3:
        dataset_count = 0
        if os.path.exists('./drawings/spiral'):
            dataset_count += 1
        if os.path.exists('./drawings/wave'):
            dataset_count += 1
        st.metric("Datasets Ready", dataset_count, "out of 2")
    
    with col4:
        results_count = len([f for f in os.listdir('./results') if f.endswith('.png')]) if os.path.exists('./results') else 0
        st.metric("Results Generated", results_count)
    
    st.markdown("##")
    
    # Getting started
    with st.expander("🚀 Getting Started Guide"):
        st.markdown("""
        ### Step 1: Setup Directory Structure
        Go to **Settings** page and click "Setup Directories" to create the required folder structure.
        
        ### Step 2: Add Your Dataset
        Place your drawing images in the appropriate folders:
        - `./drawings/spiral/training/healthy/` - Healthy spiral drawings for training
        - `./drawings/spiral/training/parkinson/` - Parkinson's spiral drawings for training
        - `./drawings/spiral/testing/healthy/` - Healthy spiral drawings for testing
        - `./drawings/spiral/testing/parkinson/` - Parkinson's spiral drawings for testing
        - Similar structure for `wave` dataset
        
        ### Step 3: Train Models
        Go to **Train Models** page and train models on your datasets.
        
        ### Step 4: Make Predictions
        Go to **Predict** page, upload an image, and get instant results!
        """)

# ==================== PREDICT PAGE ====================
elif page == "📊 Predict":
    st.markdown('<div class="main-header">📊 Image Prediction</div>', unsafe_allow_html=True)
    
    # Check if any models are available
    available_models = get_available_models()
    total_models = len(available_models['spiral']) + len(available_models['wave'])
    
    if total_models == 0:
        st.warning("⚠️ No trained models found! Please train models first in the 'Train Models' page.")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">Upload Drawing</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a spiral or wave drawing image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="Upload a clear image of a spiral or wave drawing"
        )
    
    with col2:
        st.markdown('<div class="sub-header">Settings</div>', unsafe_allow_html=True)
        
        dataset_type = st.selectbox(
            "Dataset Type",
            ["spiral", "wave"],
            help="Select the type of drawing you're uploading"
        )
        
        # Only show models that are available for the selected dataset
        available_for_dataset = available_models[dataset_type]
        if not available_for_dataset:
            st.error(f"No trained models available for {dataset_type} dataset!")
            st.stop()
        
        model_mapping = {'RF': 'Random Forest', 'SVM': 'Support Vector Machine', 'KNN': 'K-Nearest Neighbors'}
        model_options = [model_mapping[m] for m in available_for_dataset]
        
        model_display = st.selectbox(
            "Select Model",
            model_options,
            help="Choose which ML model to use for prediction"
        )
        
        # Reverse mapping
        model_name = [k for k, v in model_mapping.items() if v == model_display][0]
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        # Predict button
        if st.button("🔍 Analyze Drawing", type="primary"):
            with st.spinner("Analyzing drawing... Please wait..."):
                # Save uploaded file temporarily
                temp_path = f"./temp_upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                image.save(temp_path)
                
                try:
                    # Make prediction
                    result_img, label, prob = st.session_state.system.predict_single_image(
                        temp_path,
                        dataset_name=dataset_type,
                        model_name=model_name
                    )
                    
                    if result_img is not None:
                        # Store in history
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'dataset': dataset_type,
                            'model': model_display,
                            'prediction': label,
                            'confidence': prob
                        })
                        
                        add_log(f"Prediction: {label} ({dataset_type}/{model_name})", "success")
                        
                        with col2:
                            st.markdown("### 🎯 Prediction Result")
                            
                            # Display result image
                            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            st.image(result_img_rgb, use_container_width=True)
                        
                        # Display metrics
                        st.markdown("##")
                        st.markdown('<div class="sub-header">📊 Analysis Results</div>', unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            result_color = "🟢" if label == "Healthy" else "🔴"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{result_color} Prediction</h3>
                                <h1>{label}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if prob is not None:
                                confidence_pct = prob * 100 if label == "Parkinson's" else (1 - prob) * 100
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>📈 Confidence</h3>
                                    <h1>{confidence_pct:.1f}%</h1>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>🤖 Model Used</h3>
                                <h1>{model_display}</h1>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional information
                        st.markdown("##")
                        with st.expander("ℹ️ Understanding the Results"):
                            st.markdown(f"""
                            ### Prediction: {label}
                            
                            **Model:** {model_display}  
                            **Dataset Type:** {dataset_type.capitalize()}  
                            **Confidence Score:** {confidence_pct:.2f}%
                            
                            #### What does this mean?
                            {'- The drawing patterns suggest **healthy** motor control.' if label == 'Healthy' else "- The drawing patterns show characteristics associated with **Parkinson's disease**."}
                            - The confidence score indicates how certain the model is about this prediction.
                            - Higher confidence (>80%) indicates stronger pattern recognition.
                            
                            #### Important Note
                            ⚠️ This is a screening tool and should NOT replace professional medical diagnosis. 
                            Please consult with a healthcare professional for proper evaluation.
                            """)
                        
                        st.success("✅ Analysis complete!")
                    
                    else:
                        st.error("❌ Prediction failed. Please check the logs.")
                        add_log("Prediction failed", "error")
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    add_log(f"Error: {str(e)}", "error")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("##")
        st.markdown('<div class="sub-header">📜 Recent Predictions</div>', unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.prediction_history)
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df['confidence'] = df['confidence'].apply(lambda x: f"{x*100:.1f}%" if x is not None else "N/A")
        
        st.dataframe(
            df[['timestamp', 'dataset', 'model', 'prediction', 'confidence']],
            use_container_width=True,
            hide_index=True
        )

# ==================== TRAIN MODELS PAGE ====================
elif page == "🎯 Train Models":
    st.markdown('<div class="main-header">🎯 Model Training</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">Dataset Selection</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_spiral = st.checkbox("Train Spiral Models", value=True)
    with col2:
        train_wave = st.checkbox("Train Wave Models", value=True)
    
    if not train_spiral and not train_wave:
        st.warning("⚠️ Please select at least one dataset to train.")
    else:
        st.markdown("##")
        st.info("🔍 Training will use Random Forest, SVM, and KNN models on the selected datasets.")
        
        if st.button("🚀 Start Training", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Training logic
                datasets_to_train = []
                if train_spiral:
                    datasets_to_train.append('spiral')
                if train_wave:
                    datasets_to_train.append('wave')
                
                total_steps = len(datasets_to_train) * 3  # 3 models per dataset
                current_step = 0
                
                for dataset in datasets_to_train:
                    status_text.text(f"Processing {dataset} dataset...")
                    add_log(f"Starting training on {dataset} dataset", "info")
                    
                    # Validate dataset
                    if not st.session_state.system.validate_dataset_path(dataset):
                        st.error(f"❌ {dataset.capitalize()} dataset validation failed!")
                        add_log(f"{dataset} dataset validation failed", "error")
                        continue
                    
                    # Load and split data
                    data = st.session_state.system.select_and_split_dataset(dataset)
                    
                    if data[0] is None:
                        st.error(f"❌ Failed to load {dataset} dataset!")
                        add_log(f"Failed to load {dataset} dataset", "error")
                        continue
                    
                    trainX, trainY, testX, testY = data
                    
                    # Train models
                    for model_name, model in st.session_state.system.models.items():
                        status_text.text(f"Training {model_name} on {dataset}...")
                        
                        result = st.session_state.system.train_and_evaluate_model(
                            model_name, model, trainX, trainY, testX, testY, dataset
                        )
                        
                        if result:
                            st.session_state.system.results[dataset][model_name] = result
                            add_log(f"{model_name} trained on {dataset}: {result['accuracy']:.4f}", "success")
                        
                        current_step += 1
                        progress_bar.progress(current_step / total_steps)
                
                # Save summary
                with open('./results/model_performance_summary.txt', 'w') as f:
                    f.write("Parkinson's Disease Detection - Model Performance Summary\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for dataset in datasets_to_train:
                        if dataset in st.session_state.system.results and st.session_state.system.results[dataset]:
                            f.write(f"\n{dataset.upper()} DATASET RESULTS\n")
                            f.write("-" * 30 + "\n\n")
                            
                            best_acc = 0
                            best_model = None
                            
                            for model_name, result in st.session_state.system.results[dataset].items():
                                f.write(f"{model_name} Model: Accuracy = {result['accuracy']:.4f}\n")
                                if result['accuracy'] > best_acc:
                                    best_acc = result['accuracy']
                                    best_model = model_name
                            
                            f.write(f"\nBest model for {dataset}: {best_model} (Accuracy: {best_acc:.4f})\n")
                
                status_text.text("Training complete!")
                progress_bar.progress(1.0)
                st.success("✅ Training completed successfully!")
                add_log("Training completed", "success")
                
                # Display results
                st.markdown("##")
                st.markdown('<div class="sub-header">📊 Training Results</div>', unsafe_allow_html=True)
                
                for dataset in datasets_to_train:
                    if dataset in st.session_state.system.results and st.session_state.system.results[dataset]:
                        st.markdown(f"### {dataset.capitalize()} Dataset")
                        
                        cols = st.columns(3)
                        for idx, (model_name, result) in enumerate(st.session_state.system.results[dataset].items()):
                            with cols[idx]:
                                accuracy_pct = result['accuracy'] * 100
                                st.metric(
                                    model_name,
                                    f"{accuracy_pct:.2f}%",
                                    delta="Accuracy"
                                )
                        
                        # Show confusion matrices
                        st.markdown("#### Confusion Matrices")
                        cm_cols = st.columns(3)
                        for idx, model_name in enumerate(['RF', 'SVM', 'KNN']):
                            cm_path = f'./results/{dataset}_{model_name}_confusion_matrix.png'
                            if os.path.exists(cm_path):
                                with cm_cols[idx]:
                                    st.image(cm_path, caption=f"{model_name} Confusion Matrix", use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                add_log(f"Training error: {str(e)}", "error")

# ==================== COMPARE RESULTS PAGE ====================
elif page == "📈 Compare Results":
    st.markdown('<div class="main-header">📈 Model Comparison</div>', unsafe_allow_html=True)
    
    # Check if results exist
    if not os.path.exists('./results/model_performance_summary.txt'):
        st.warning("⚠️ No training results found. Please train models first.")
        st.stop()
    
    # Load results
    results_data = {'spiral': {}, 'wave': {}}
    
    for dataset in ['spiral', 'wave']:
        for model in ['RF', 'SVM', 'KNN']:
            model_path = f'./models/{dataset}/{model}_model.pkl'
            if os.path.exists(model_path):
                # Try to get accuracy from results
                if dataset in st.session_state.system.results and model in st.session_state.system.results[dataset]:
                    results_data[dataset][model] = st.session_state.system.results[dataset][model]['accuracy']
    
    # Display comparison
    st.markdown('<div class="sub-header">Accuracy Comparison</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📊 Charts", "📋 Details"])
    
    with tab1:
        # Create comparison charts
        if results_data['spiral'] or results_data['wave']:
            col1, col2 = st.columns(2)
            
            with col1:
                if results_data['spiral']:
                    st.markdown("#### Spiral Dataset")
                    models = list(results_data['spiral'].keys())
                    accuracies = [results_data['spiral'][m] * 100 for m in models]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=models, y=accuracies, marker_color='#1f77b4')
                    ])
                    fig.update_layout(
                        title="Model Accuracy Comparison",
                        xaxis_title="Model",
                        yaxis_title="Accuracy (%)",
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if results_data['wave']:
                    st.markdown("#### Wave Dataset")
                    models = list(results_data['wave'].keys())
                    accuracies = [results_data['wave'][m] * 100 for m in models]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=models, y=accuracies, marker_color='#ff7f0e')
                    ])
                    fig.update_layout(
                        title="Model Accuracy Comparison",
                        xaxis_title="Model",
                        yaxis_title="Accuracy (%)",
                        yaxis_range=[0, 100]
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No model results available for comparison.")
    
    with tab2:
        # Display confusion matrices
        for dataset in ['spiral', 'wave']:
            if results_data[dataset]:
                st.markdown(f"### {dataset.capitalize()} Dataset Results")
                
                cols = st.columns(3)
                for idx, model in enumerate(['RF', 'SVM', 'KNN']):
                    cm_path = f'./results/{dataset}_{model}_confusion_matrix.png'
                    if os.path.exists(cm_path):
                        with cols[idx]:
                            st.image(cm_path, caption=f"{model}", use_container_width=True)
                
                st.markdown("---")

# ==================== VISUALIZATION PAGE ====================
elif page == "🔬 Visualization":
    st.markdown('<div class="main-header">🔬 Preprocessing Visualization</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload an image to visualize the preprocessing steps used in the detection system.
    This helps understand how the system prepares images before analysis.
    """)
    
    uploaded_file = st.file_uploader(
        "Choose an image to visualize preprocessing",
        type=['png', 'jpg', 'jpeg', 'bmp']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        if st.button("🔍 Visualize Preprocessing", type="primary"):
            with st.spinner("Processing..."):
                # Save temp file
                temp_path = f"./temp_viz_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                image.save(temp_path)
                
                try:
                    st.session_state.system.visualize_preprocessing(temp_path)
                    
                    # Display result
                    result_path = f"./results/preprocessing_{os.path.basename(temp_path)}.png"
                    if os.path.exists(result_path):
                        with col2:
                            st.markdown("### Preprocessing Steps")
                            st.image(result_path, use_container_width=True)
                        
                        add_log("Preprocessing visualization generated", "success")
                        st.success("✅ Preprocessing visualization complete!")
                        
                        with st.expander("ℹ️ Understanding the Preprocessing Steps"):
                            st.markdown("""
                            ### Preprocessing Pipeline
                            
                            1. **Original Image**: The raw input image
                            2. **Grayscale Conversion**: Converts to single channel for processing
                            3. **Binary Threshold (Otsu)**: Automatic threshold to separate foreground/background
                            
                            These steps help extract relevant features while removing noise and normalizing the image.
                            """)
                    else:
                        st.error("Failed to generate visualization.")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    add_log(f"Visualization error: {str(e)}", "error")
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# ==================== SETTINGS PAGE ====================
elif page == "⚙️ Settings":
    st.markdown('<div class="main-header">⚙️ System Settings</div>', unsafe_allow_html=True)
    
    # Directory setup
    st.markdown('<div class="sub-header">📁 Directory Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗂️ Setup Directory Structure", type="primary"):
            try:
                create_sample_dirs()
                st.success("✅ Directory structure created successfully!")
                add_log("Directory structure created", "success")
                
                st.info("""
                Created directory structure:
                - `./drawings/spiral/training/` (healthy & parkinson)
                - `./drawings/spiral/testing/` (healthy & parkinson)
                - `./drawings/wave/training/` (healthy & parkinson)
                - `./drawings/wave/testing/` (healthy & parkinson)
                - `./models/`
                - `./results/`
                """)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                add_log(f"Directory setup error: {str(e)}", "error")
    
    with col2:
        if st.button("📊 Check Dataset Status"):
            st.markdown("### Dataset Status")
            for dataset in ['spiral', 'wave']:
                path = Path(f'./drawings/{dataset}')
                if path.exists():
                    train_imgs = len(list((path / 'training').rglob('*.png'))) + \
                                len(list((path / 'training').rglob('*.jpg')))
                    test_imgs = len(list((path / 'testing').rglob('*.png'))) + \
                               len(list((path / 'testing').rglob('*.jpg')))
                    
                    st.metric(
                        f"{dataset.capitalize()} Dataset",
                        f"{train_imgs + test_imgs} images",
                        delta=f"Train: {train_imgs}, Test: {test_imgs}"
                    )
                else:
                    st.warning(f"{dataset.capitalize()} dataset not found")
    
    st.markdown("##")
    
    # Model management
    st.markdown('<div class="sub-header">🤖 Model Management</div>', unsafe_allow_html=True)
    
    available = get_available_models()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Spiral Models")
        if available['spiral']:
            for model in available['spiral']:
                st.success(f"✅ {model} model trained")
        else:
            st.info("No spiral models trained yet")
    
    with col2:
        st.markdown("#### Wave Models")
        if available['wave']:
            for model in available['wave']:
                st.success(f"✅ {model} model trained")
        else:
            st.info("No wave models trained yet")
    
    st.markdown("##")
    
    # Clear data
    st.markdown('<div class="sub-header">🗑️ Data Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Clear Prediction History"):
            st.session_state.prediction_history = []
            st.success("Prediction history cleared!")
    
    with col2:
        if st.button("📋 Clear Activity Logs"):
            st.session_state.logs = []
            st.success("Activity logs cleared!")
    
    st.markdown("##")
    
    # System info
    with st.expander("💻 System Information"):
        st.markdown(f"""
        **Python Version:** {sys.version}  
        **Working Directory:** {os.getcwd()}  
        **Models Directory:** `./models/`  
        **Results Directory:** `./results/`  
        **Drawings Directory:** `./drawings/`  
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 Parkinson's Disease Detection System | Built with Streamlit & Machine Learning</p>
    <p style='font-size: 0.8rem;'>⚠️ This tool is for screening purposes only and should not replace professional medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)