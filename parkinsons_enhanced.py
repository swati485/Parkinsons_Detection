# Parkinson's Disease Detection System
# A complete implementation for processing spiral and wave drawings
# to detect Parkinson's disease through machine learning

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from imutils import paths
from skimage import feature
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle
import matplotlib.pyplot as plt
import argparse

class ParkinsonsDetectionSystem:
    def __init__(self):
        """Initialize the Parkinson's Disease Detection System"""
        # Create necessary directories
        os.makedirs('./models/spiral', exist_ok=True)
        os.makedirs('./models/wave', exist_ok=True)
        os.makedirs('./results', exist_ok=True)
        os.makedirs('./drawings', exist_ok=True)
        
        # Define classifier models
        self.models = {
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        self.results = {
            'spiral': {},
            'wave': {}
        }

    def validate_dataset_path(self, dataset_name):
        """Validate that the dataset path exists and contains data"""
        path = Path(f'./drawings/{dataset_name}')
        
        if not path.exists():
            print(f"ERROR: Dataset path '{path}' does not exist.")
            print(f"Please ensure you have a folder named '{dataset_name}' in the './drawings' directory.")
            return False
        
        training_path = path / 'training'
        testing_path = path / 'testing'
        
        if not training_path.exists() or not testing_path.exists():
            print(f"ERROR: Missing 'training' or 'testing' subdirectories in '{path}'")
            print("Please ensure your dataset has the correct structure.")
            return False
        
        # Check if there are images
        training_images = list(paths.list_images(str(training_path)))
        testing_images = list(paths.list_images(str(testing_path)))
        
        if len(training_images) == 0 or len(testing_images) == 0:
            print(f"ERROR: No images found in training or testing directories.")
            print(f"Training images count: {len(training_images)}")
            print(f"Testing images count: {len(testing_images)}")
            return False
        
        # All checks passed
        print(f"Dataset validation successful for '{dataset_name}'")
        print(f"Found {len(training_images)} training images and {len(testing_images)} testing images")
        return True

    def extract_hog_features(self, image):
        """Extract HOG features with appropriate parameters"""
        features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
        return features

    def preprocess_image(self, image):
        """Basic preprocessing pipeline"""
        # Resize
        image = cv2.resize(image, (200, 200))
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold with Otsu's method
        _, image_binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return image_binary

    def process_dataset(self, path):
        """Process the dataset with error handling"""
        print(f"Processing images from: {path}")
        imagePaths = list(paths.list_images(path))
        
        if len(imagePaths) == 0:
            print(f"WARNING: No images found in {path}")
            return np.array([]), np.array([])
        
        data = []
        labels = []
        
        for i, imagePath in enumerate(imagePaths):
            try:
                # Print progress every 10 images
                if i % 10 == 0:
                    print(f"Processing image {i+1}/{len(imagePaths)}: {imagePath}")
                    
                # Extract label from path
                label = imagePath.split(os.path.sep)[-2]
                
                # Read and process image
                image = cv2.imread(imagePath)
                if image is None:
                    print(f"WARNING: Could not read image: {imagePath}")
                    continue
                    
                # Process image
                processed_image = self.preprocess_image(image)
                
                # Extract features
                features = self.extract_hog_features(processed_image)
                
                # Append to lists
                data.append(features)
                labels.append(label)
            except Exception as e:
                print(f"ERROR processing {imagePath}: {str(e)}")
        
        if len(data) == 0:
            print("WARNING: No valid data was extracted from images")
            return np.array([]), np.array([])
        
        print(f"Successfully processed {len(data)} images")
        return np.array(data), np.array(labels)

    def select_and_split_dataset(self, data_set):
        """Process dataset with proper error handling"""
        print(f"\nProcessing {data_set} dataset...")
        
        # Validate dataset path
        if not self.validate_dataset_path(data_set):
            print(f"Skipping {data_set} dataset due to validation errors")
            return None, None, None, None
        
        # Define paths
        path = f'./drawings/{data_set}'
        trainingPath = os.path.join(path, "training")
        testingPath = os.path.join(path, "testing")
        
        # Load the data
        trainX, trainY = self.process_dataset(trainingPath)
        testX, testY = self.process_dataset(testingPath)
        
        # Check if data was loaded successfully
        if trainX.size == 0 or testX.size == 0:
            print(f"ERROR: Failed to load {data_set} dataset")
            return None, None, None, None
        
        # Print dataset statistics
        print(f"\nDataset statistics for {data_set}:")
        print(f"Training set: {trainX.shape[0]} samples, {trainX.shape[1]} features")
        print(f"Testing set: {testX.shape[0]} samples, {testX.shape[1]} features")
        
        # Check for class balance
        unique_train, counts_train = np.unique(trainY, return_counts=True)
        unique_test, counts_test = np.unique(testY, return_counts=True)
        
        print("\nTraining set class distribution:")
        for cls, count in zip(unique_train, counts_train):
            print(f"  {cls}: {count} samples")
        
        print("\nTesting set class distribution:")
        for cls, count in zip(unique_test, counts_test):
            print(f"  {cls}: {count} samples")
        
        # Standardize features
        print("\nStandardizing features...")
        scaler = StandardScaler()
        trainX = scaler.fit_transform(trainX)
        testX = scaler.transform(testX)
        
        # Encode labels
        print("Encoding labels...")
        le = LabelEncoder()
        trainY = le.fit_transform(trainY)
        testY = le.transform(testY)
        
        # Save the scaler and encoder
        print(f"Saving scaler and encoder to './models/{data_set}/'")
        with open(f'./models/{data_set}/{data_set}_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(f'./models/{data_set}/{data_set}_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        
        return trainX, trainY, testX, testY

    def train_and_evaluate_model(self, model_name, model, trainX, trainY, testX, testY, dataset_name):
        """Train and evaluate a model with proper error handling"""
        print(f"\nTraining {model_name} model for {dataset_name} dataset...")
        
        try:
            # Train the model
            model.fit(trainX, trainY)
            
            # Save the model
            print(f"Saving model to './models/{dataset_name}/{model_name}_model.pkl'")
            with open(f'./models/{dataset_name}/{model_name}_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            
            # Evaluate the model
            print(f"Evaluating {model_name} model...")
            pred = model.predict(testX)
            
            # Calculate metrics
            accuracy = accuracy_score(testY, pred)
            cm = confusion_matrix(testY, pred)
            report = classification_report(testY, pred)
            
            # Print results
            print(f"\nResults for {model_name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Confusion Matrix:")
            print(cm)
            print("Classification Report:")
            print(report)
            
            # Save confusion matrix visualization
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {model_name} on {dataset_name}')
            plt.colorbar()
            classes = ['Healthy', 'Parkinsons']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(f'./results/{dataset_name}_{model_name}_confusion_matrix.png')
            plt.close()
            
            return {
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': pred
            }
        
        except Exception as e:
            print(f"ERROR training/evaluating {model_name} model: {str(e)}")
            return None

    def predict_single_image(self, image_path, dataset_name='spiral', model_name='RF'):
        """Make a prediction for a single image with error handling"""
        print(f"Predicting image: {image_path}")
        
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"ERROR: Image file not found: {image_path}")
                return None, None, None
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"ERROR: Could not read image: {image_path}")
                return None, None, None
            
            # Check if model exists
            model_path = f'./models/{dataset_name}/{model_name}_model.pkl'
            scaler_path = f'./models/{dataset_name}/{dataset_name}_scaler.pkl'
            encoder_path = f'./models/{dataset_name}/{dataset_name}_encoder.pkl'
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path]):
                print("ERROR: Model files not found. Make sure you've trained the models first.")
                return None, None, None
            
            # Load the scaler, encoder, and model
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                le = pickle.load(f)
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            # Extract features
            features = self.extract_hog_features(processed_image)
            
            # Scale features
            scaled_features = scaler.transform([features])
            
            # Make prediction
            pred = model.predict(scaled_features)
            
            # Get probability if available
            prob = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(scaled_features)[0][1]
            
            # Get class label
            label = 'Parkinson\'s' if pred[0] == 1 else 'Healthy'
            
            # Create output image
            result_img = image.copy()
            result_img = cv2.resize(result_img, (400, 400))
            
            # Add text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (0, 255, 0) if pred[0] == 0 else (0, 0, 255)  # Green for healthy, red for Parkinson's
            confidence_text = f" (Confidence: {prob:.2f})" if prob is not None else ""
            cv2.putText(result_img, label + confidence_text, (10, 30), font, 0.7, color, 2)
            
            # Save the result image
            result_filename = f"./results/prediction_{os.path.basename(image_path)}"
            cv2.imwrite(result_filename, result_img)
            
            print(f"Prediction: {label}{confidence_text}")
            print(f"Result image saved to {result_filename}")
            
            return result_img, label, prob
        
        except Exception as e:
            print(f"ERROR in prediction: {str(e)}")
            return None, None, None
    
    def visualize_preprocessing(self, image_path):
        """Visualize the preprocessing steps for educational purposes"""
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"ERROR: Image file not found: {image_path}")
                return
            
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"ERROR: Could not read image: {image_path}")
                return
            
            # Create figure with subplots
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            
            # Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            axs[1].imshow(gray, cmap='gray')
            axs[1].set_title('Grayscale')
            axs[1].axis('off')
            
            # Binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            axs[2].imshow(binary, cmap='gray')
            axs[2].set_title('Binary Threshold (Otsu)')
            axs[2].axis('off')
            
            plt.tight_layout()
            
            # Save the visualization
            output_path = f"./results/preprocessing_{os.path.basename(image_path)}.png"
            plt.savefig(output_path)
            plt.close()
            
            print(f"Preprocessing visualization saved to {output_path}")
            
        except Exception as e:
            print(f"ERROR in preprocessing visualization: {str(e)}")

    def train_all_models(self):
        """Train all models for both datasets"""
        # Process spiral dataset
        spiral_data = self.select_and_split_dataset('spiral')
        
        if spiral_data[0] is not None:
            spiral_trainX, spiral_trainY, spiral_testX, spiral_testY = spiral_data
            
            # Train and evaluate models for spiral dataset
            for name, model in self.models.items():
                result = self.train_and_evaluate_model(name, model, 
                                                 spiral_trainX, spiral_trainY, 
                                                 spiral_testX, spiral_testY, 'spiral')
                if result:
                    self.results['spiral'][name] = result
        
        # Process wave dataset
        wave_data = self.select_and_split_dataset('wave')
        
        if wave_data[0] is not None:
            wave_trainX, wave_trainY, wave_testX, wave_testY = wave_data
            
            # Train and evaluate models for wave dataset
            for name, model in self.models.items():
                result = self.train_and_evaluate_model(name, model, 
                                                 wave_trainX, wave_trainY, 
                                                 wave_testX, wave_testY, 'wave')
                if result:
                    self.results['wave'][name] = result
        
        # Save overall results summary
        with open('./results/model_performance_summary.txt', 'w') as f:
            f.write("Parkinson's Disease Detection - Model Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for dataset in ['spiral', 'wave']:
                if dataset in self.results and self.results[dataset]:
                    f.write(f"\n{dataset.upper()} DATASET RESULTS\n")
                    f.write("-" * 30 + "\n\n")
                    
                    # Find best model
                    best_acc = 0
                    best_model = None
                    
                    for model_name, result in self.results[dataset].items():
                        f.write(f"{model_name} Model: Accuracy = {result['accuracy']:.4f}\n")
                        if result['accuracy'] > best_acc:
                            best_acc = result['accuracy']
                            best_model = model_name
                    
                    f.write(f"\nBest model for {dataset}: {best_model} (Accuracy: {best_acc:.4f})\n")
            
        print("\nTraining and evaluation complete!")
        print("Results summary saved to './results/model_performance_summary.txt'")

def create_sample_dirs():
    """Create sample directory structure for users to understand expected format"""
    sample_dirs = [
        './drawings/spiral/training/healthy',
        './drawings/spiral/training/parkinson',
        './drawings/spiral/testing/healthy',
        './drawings/spiral/testing/parkinson',
        './drawings/wave/training/healthy',
        './drawings/wave/training/parkinson',
        './drawings/wave/testing/healthy',
        './drawings/wave/testing/parkinson'
    ]
    
    for d in sample_dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create a README file explaining the directory structure
    with open('./drawings/README.txt', 'w') as f:
        f.write("""
Parkinson's Disease Detection System - Dataset Structure

This folder should contain your drawing images organized as follows:

./drawings/
├── spiral/
│   ├── training/
│   │   ├── healthy/       (Place healthy spiral drawings for training here)
│   │   └── parkinson/     (Place Parkinson's spiral drawings for training here)
│   └── testing/
│       ├── healthy/       (Place healthy spiral drawings for testing here)
│       └── parkinson/     (Place Parkinson's spiral drawings for testing here)
└── wave/
    ├── training/
    │   ├── healthy/       (Place healthy wave drawings for training here)
    │   └── parkinson/     (Place Parkinson's wave drawings for training here)
    └── testing/
        ├── healthy/       (Place healthy wave drawings for testing here)
        └── parkinson/     (Place Parkinson's wave drawings for testing here)

Images can be in any common format (jpg, png, etc.)
        """)
    
    print("Created sample directory structure in './drawings/'")
    print("Check './drawings/README.txt' for guidance on organizing your dataset")

def main():
    """Main function to handle command-line arguments and program flow"""
    parser = argparse.ArgumentParser(description="Parkinson's Disease Detection System")
    
    # Define command line arguments
    parser.add_argument('--action', type=str, default='train',
                        choices=['train', 'predict', 'visualize', 'setup'],
                        help='Action to perform: train models, predict on image, visualize preprocessing, or setup directories')
    
    parser.add_argument('--image', type=str,
                        help='Path to image for prediction or visualization')
    
    parser.add_argument('--dataset', type=str, default='spiral',
                        choices=['spiral', 'wave'],
                        help='Dataset to use for prediction (spiral or wave)')
    
    parser.add_argument('--model', type=str, default='RF',
                        choices=['RF', 'SVM', 'KNN'],
                        help='Model to use for prediction')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create system instance
    system = ParkinsonsDetectionSystem()
    
    # Perform requested action
    if args.action == 'setup':
        create_sample_dirs()
    
    elif args.action == 'train':
        print("Parkinson's Disease Detection System - Training Mode")
        print("=" * 50)
        system.train_all_models()
    
    elif args.action == 'predict':
        if not args.image:
            print("ERROR: --image parameter is required for prediction")
            return
        
        print(f"Predicting image using {args.model} model on {args.dataset} dataset")
        result_img, label, prob = system.predict_single_image(
            args.image, 
            dataset_name=args.dataset,
            model_name=args.model
        )
        
        if result_img is not None:
            # Display if running in an environment with GUI
            try:
                cv2.imshow("Prediction Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                pass
    
    elif args.action == 'visualize':
        if not args.image:
            print("ERROR: --image parameter is required for visualization")
            return
        
        print("Visualizing preprocessing steps")
        system.visualize_preprocessing(args.image)
    
    else:
        print("Invalid action. Use --help for usage information.")

if __name__ == "__main__":
    print("\nParkinson's Disease Detection System")
    print("===================================\n")
    main()