import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Enable mixed precision
keras.mixed_precision.set_global_policy('mixed_float16')

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
            "VGG16": keras.applications.VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet'),
            "ResNet50": keras.applications.ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet'),
            "MobileNetV2": keras.applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet'),
            "DenseNet121": keras.applications.DenseNet121(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
        }

    def load_dataset(self):
        # Load CIFAR-10 dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

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
        self.y_train = keras.utils.to_categorical(self.y_train, 10)  # One-hot encode labels
        self.y_test = keras.utils.to_categorical(self.y_test, 10)

    @staticmethod
    def apply_augmentations(x, y):
        """Apply augmentations to the dataset."""

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            zoom_range=0.2,
            fill_mode='nearest'
        )
        augmented_data = datagen.flow(x, y, batch_size=len(x), shuffle=False)
        x_augmented, y_augmented = next(augmented_data)
        return x_augmented, y_augmented

    def start(self, epochs=10, batch_size=32):
        for model_name, base_model in self.models.items():
            # Train and evaluate on original data
            self.prepare_pretrained_model(model_name, base_model, epochs=3 if self.is_demo else epochs, batch_size=batch_size, augmented=False)
            # Train and evaluate on augmented data
            self.prepare_pretrained_model(model_name, base_model, epochs=3 if self.is_demo else epochs, batch_size=batch_size, augmented=True)

        self.save_results()
        self.display_results()

    def save_results(self):
        """Save the results to a CSV file."""
        os.makedirs(self.results_path, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.results_path, "results.csv"), index=False)

    @staticmethod
    def extract_small_batch(x, y, size):
        """Return a small subset of the data."""
        idx = np.random.choice(len(x), size, replace=False)  # Randomly select indices
        small_x, small_y = x[idx], y[idx]
        return x[idx], y[idx].reshape(-1, 1)  # Ensure y retains the shape (n_samples, 1)

    def train_model(self, model, epochs=10, batch_size=64):
        """Train the given model."""
        self.model = model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_ds, epochs=epochs, batch_size=batch_size, validation_data=self.test_ds, verbose=2)

    def evaluate_model(self, model_name, augmented):
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
        print(f"F1 Score: {f1:.4f}")

        # Save results
        self.results.append({
            "Model": model_name + ("_augmented" if augmented else "_original"),
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

        # Clear the Keras session to free up memory
        keras.backend.clear_session()

    def prepare_pretrained_model(self, model_name, base_model=None, epochs=5, batch_size=64, augmented=False):
        """Prepare and train a pre-trained model."""
        if model_name == "CNN":
            model = keras.models.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(128, (3, 3), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation='softmax')
            ])
        else:
            model = keras.models.Sequential([
                base_model,
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation='softmax')
            ])

        if augmented:
            x_train, y_train = self.apply_augmentations(self.x_train, self.y_train)
        else:
            x_train, y_train = self.x_train, self.y_train


        # Create data generators
        self.train_ds = ImageDataGenerator().flow(x_train, y_train, batch_size=batch_size)
        self.test_ds = ImageDataGenerator().flow(self.x_test, self.y_test, batch_size=batch_size)

        print(f"\nTraining {model_name} Model ({'Augmented' if augmented else 'Original'})...")
        self.train_model(model, epochs=epochs, batch_size=batch_size)
        print(f"\nEvaluating {model_name} Model ({'Augmented' if augmented else 'Original'})...")
        self.evaluate_model(model_name, augmented)

    def display_results(self):
        """Display the results in a table."""
        # load the results from the CSV file
        df = pd.read_csv(os.path.join(self.results_path, "results.csv"))
        print(df.to_string(index=False))

        # Plot Test Accuracy
        plt.figure(figsize=(10, 6))
        plt.bar(df['Model'], df['Test Accuracy'], color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy of Different Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Plot Test Loss
        plt.figure(figsize=(10, 6))
        plt.bar(df['Model'], df['Test Loss'], color='salmon')
        plt.xlabel('Model')
        plt.ylabel('Test Loss')
        plt.title('Test Loss of Different Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Plot F1 Score
        plt.figure(figsize=(10, 6))
        plt.bar(df['Model'], df['F1 Score'], color='lightgreen')
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        plt.title('F1 Score of Different Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # Line plot for all metrics
        plt.figure(figsize=(12, 8))
        plt.plot(df['Model'], df['Test Accuracy'], marker='o', label='Test Accuracy', color='skyblue')
        plt.plot(df['Model'], df['Test Loss'], marker='o', label='Test Loss', color='salmon')
        plt.plot(df['Model'], df['F1 Score'], marker='o', label='F1 Score', color='lightgreen')
        plt.xlabel('Model')
        plt.ylabel('Metrics')
        plt.title('Comparison of Different Models')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    running_mode = input("Do you want to run in demo mode? (yes/no): ").strip().lower()

    # Create a ModelTrainer instance
    trainer = ModelTrainer(is_demo=running_mode == 'yes', results_path="results")
    trainer.start(epochs=20)