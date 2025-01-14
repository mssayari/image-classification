import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, InceptionV3, DenseNet121

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class ModelTrainer:
    def __init__(self, is_demo=False, results_path="results"):
        self.is_demo = is_demo
        self.model = None
        self.results = []
        self.train_ds = None
        self.test_ds = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.models = {}
        self.results_path = results_path

        self.initialize_models()
        self.load_dataset()

    def initialize_models(self):
        """Initialize all models with their respective base models."""
        self.models = {
            "CNN": None,
            # "VGG16": VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet'),
            # "ResNet50": ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet'),
            # "MobileNetV2": MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet'),
            # "DenseNet121": DenseNet121(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
        }

    def load_dataset(self):
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        if self.is_demo:
            # Demo mode: Use a small subset of the dataset
            print("Running in demo mode with a small dataset...")
            self.x_train, self.y_train = self.extract_small_batch(x_train, y_train, size=5000)
            self.x_test, self.y_test = self.extract_small_batch(x_test, y_test, size=1000)
        else:
            # Full mode: Use the entire dataset
            print("Running in full mode with the complete dataset...")
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test

        self.x_train = self.x_train / 255.0  # Normalize pixel values
        self.x_test = self.x_test / 255.0
        self.y_train = to_categorical(self.y_train, 10)  # One-hot encode labels
        self.y_test = to_categorical(self.y_test, 10)

    def start(self):
        for model_name, base_model in self.models.items():
            self.prepare_pretrained_model(model_name, base_model, epochs=3 if self.is_demo else 10, batch_size=32)

        self.save_results()
        self.display_results()

    def save_results(self):
        """Save the results to a CSV file."""
        os.makedirs(self.results_path, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.results_path, "results.csv"), index=False)
    def extract_small_batch(self, x, y, size):
        """Return a small subset of the data."""
        idx = np.random.choice(len(x), size, replace=False)  # Randomly select indices
        small_x, small_y = x[idx], y[idx]
        return x[idx], y[idx].reshape(-1, 1)  # Ensure y retains the shape (n_samples, 1)

    def train_model(self, model, epochs=10, batch_size=64):
        """Train the given model."""
        self.model = model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_ds, epochs=epochs, batch_size=batch_size, validation_data=self.test_ds, verbose=2)

    def evaluate_model(self, model_name):
        """Evaluate the trained model with optimized memory usage."""
        if not self.model:
            raise ValueError("No model has been trained yet.")

        # Evaluate the model using `evaluate` method
        test_loss, test_accuracy = self.model.evaluate(self.test_ds, verbose=0)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Initialize lists for predictions and true labels
        y_pred_classes = []
        y_true_classes = []

        # Predict in batches
        for batch_x, batch_y in self.test_ds:
            batch_pred = self.model.predict_on_batch(batch_x)
            y_pred_classes.extend(np.argmax(batch_pred, axis=1))  # Predicted labels
            y_true_classes.extend(np.argmax(batch_y, axis=1))  # True labels

            # Stop after processing all samples
            if len(y_true_classes) >= len(self.x_test):
                break

        y_pred_classes = np.array(y_pred_classes)
        y_true_classes = np.array(y_true_classes)

        # Compute F1 score
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

        # Save results
        self.results.append({
            "Model": model_name,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "F1 Score": f1
        })

        # Generate confusion matrix
        cm = tf.math.confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def prepare_pretrained_model(self, model_name, base_model=None, epochs=5, batch_size=64):
        """Prepare and train a pre-trained model."""
        if model_name == "CNN":
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])
        else:
            model = models.Sequential([
                base_model,
                # layers.Flatten(),
                GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])

        # Create ImageDataGenerator instances
        if model_name == "MobileNetV2" or model_name == "DenseNet121":
            train_image_generator = ImageDataGenerator()
            test_image_generator = ImageDataGenerator()
        else:
            train_image_generator = ImageDataGenerator()
            test_image_generator = ImageDataGenerator()

        # Create data generators
        self.train_ds = train_image_generator.flow(self.x_train, self.y_train, batch_size=batch_size)
        self.test_ds = test_image_generator.flow(self.x_test, self.y_test, batch_size=batch_size)

        print(f"\nTraining {model_name} Model...")
        self.train_model(model, epochs=epochs, batch_size=batch_size)
        print(f"\nEvaluating {model_name} Model...")
        self.evaluate_model(model_name)

    def display_results(self):
        """Display the results in a table."""

        # load the results from the CSV file
        df = pd.read_csv(os.path.join(self.results_path, "results.csv"))

        print(df.to_string(index=False))


if __name__ == "__main__":

    running_mode = input("Do you want to run in demo mode? (yes/no): ").strip().lower()

    # Create a ModelTrainer instance
    trainer = ModelTrainer(is_demo=running_mode == 'yes', results_path="results")
    trainer.start()