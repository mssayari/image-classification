import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

class ModelTrainer:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train / 255.0  # Normalize pixel values
        self.x_test = x_test / 255.0
        self.y_train = to_categorical(y_train, 10)  # One-hot encode labels
        self.y_test = to_categorical(y_test, 10)
        self.model = None
        self.results = []
        self.train_ds = None
        self.test_ds = None

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
        if model_name == "CustomCNN":
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

        # if model_name == "MobileNetV2" or model_name == "DenseNet121":
        #     self.resize_images(self.x_train, self.x_test, batch_size=batch_size)

        print(f"\nTraining {model_name} Model...")
        self.train_model(model, epochs=epochs, batch_size=batch_size)
        print(f"\nEvaluating {model_name} Model...")
        self.evaluate_model(model_name)

    def display_results(self):
        """Display the results in a table."""
        df = pd.DataFrame(self.results)
        print(df)


def get_small_dataset(x, y, size=1000):
    """Return a small subset of the data."""
    idx = np.random.choice(len(x), size, replace=False)  # Randomly select indices
    small_x, small_y = x[idx], y[idx]
    return x[idx], y[idx].reshape(-1, 1)  # Ensure y retains the shape (n_samples, 1)


if __name__ == "__main__":

    mode = input("Do you want to run in demo mode? (yes/no): ").strip().lower()

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    if mode == 'yes':
        # Demo mode: Use a small subset of the dataset
        print("Running in demo mode with a small dataset...")
        x_train, y_train = get_small_dataset(x_train, y_train, size=5000)
        x_test, y_test = get_small_dataset(x_test, y_test, size=1000)
    else:
        # Full mode: Use the entire dataset
        print("Running in full mode with the complete dataset...")

    # Create a ModelTrainer instance
    trainer = ModelTrainer(x_train, y_train, x_test, y_test)

    # Train and evaluate custom CNN
    trainer.prepare_pretrained_model(model_name="CustomCNN", base_model=None, epochs=3 if mode == 'yes' else 10, batch_size=32)

    # Train and evaluate pre-trained models
    vgg16_base = tf.keras.applications.VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    trainer.prepare_pretrained_model(model_name="VGG16", base_model=vgg16_base, epochs=3 if mode == 'yes' else 5)

    resnet50_base = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    trainer.prepare_pretrained_model(model_name="ResNet50", base_model=resnet50_base, epochs=3 if mode == 'yes' else 5)

    mobilenetv2_base = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet'
    )
    trainer.prepare_pretrained_model(
        model_name="MobileNetV2",
        base_model=mobilenetv2_base,
        epochs=3 if mode == 'yes' else 5,
        batch_size=32
    )

    densenet121_base = DenseNet121(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    trainer.prepare_pretrained_model(model_name="DenseNet121", base_model=densenet121_base, epochs=3 if mode == 'yes' else 5, batch_size=32)

    # Display all results
    trainer.display_results()
