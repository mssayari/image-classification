import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, f1_score
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.densenet import DenseNet121




class ModelTrainer:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train / 255.0  # Normalize pixel values
        self.x_test = x_test / 255.0
        self.y_train = to_categorical(y_train, 10)  # One-hot encode labels
        self.y_test = to_categorical(y_test, 10)
        self.model = None
        self.results = []

    def resize_images(self, x_train, x_test):
        """Resize images to the specified size."""
        x_train_resized = tf.image.resize(x_train, [224, 224]).numpy()
        x_test_resized = tf.image.resize(x_test, [224, 224]).numpy()
        x_train_resized = preprocess_input(x_train_resized)
        x_test_resized = preprocess_input(x_test_resized)
        self.x_train = x_train_resized
        self.x_test = x_test_resized

    @staticmethod
    def create_cnn_model():
        """Define a basic CNN model."""
        model = models.Sequential([
            # layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
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
        return model

    def train_model(self, model, epochs=10, batch_size=64):
        """Train the given model."""
        self.model = model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       validation_data=(self.x_test, self.y_test), verbose=2)

    def evaluate_model(self, model_name):
        """Evaluate the trained model."""
        if not self.model:
            raise ValueError("No model has been trained yet.")
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)

        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

        # Save the results
        self.results.append({
            "Model": model_name,
            "Test Loss": test_loss,
            "Test Accuracy": test_accuracy,
            "F1 Score": f1
        })

        cm = tf.math.confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def prepare_pretrained_model(self, base_model, model_name, epochs=5, batch_size=64):

        if model_name == "MobileNetV2" or model_name == "DenseNet121":
            self.resize_images(self.x_train, self.x_test)

        """Prepare and train a pre-trained model."""
        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
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
    cnn_model = ModelTrainer.create_cnn_model()
    trainer.train_model(cnn_model, epochs=3 if mode == 'yes' else 10)
    trainer.evaluate_model("Custom CNN")

    # Train and evaluate pre-trained models
    vgg16_base = tf.keras.applications.VGG16(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    trainer.prepare_pretrained_model(vgg16_base, "VGG16", epochs=3 if mode == 'yes' else 5)

    resnet50_base = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    trainer.prepare_pretrained_model(resnet50_base, "ResNet50", epochs=3 if mode == 'yes' else 5)

    mobilenetv2_base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    trainer.prepare_pretrained_model(mobilenetv2_base, "MobileNetV2", epochs=3 if mode == 'yes' else 5)

    densenet121_base = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    trainer.prepare_pretrained_model(densenet121_base, "DenseNet121", epochs=3 if mode == 'yes' else 5)

    # Display all results
    trainer.display_results()