import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


class DatasetHandler:
    def __init__(self, base_dir="./data", augmentations_per_image=5):
        self.base_dir = base_dir
        self.augmentations_per_image = augmentations_per_image
        self.train_dir = os.path.join(base_dir, "train")
        self.test_dir = os.path.join(base_dir, "test")
        self.augmented_dir = os.path.join(base_dir, "augmented")
        self.feature_dir = os.path.join(base_dir, "features")

    def download_dataset(self):
        """Downloads CIFAR-10 dataset and saves it locally in labeled folders."""
        if os.path.exists(self.train_dir) and os.path.exists(self.test_dir):
            print("Dataset already downloaded.")
            return

        print("Downloading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        print("Saving training dataset...")
        self._save_images(x_train, y_train, self.train_dir)

        print("Saving testing dataset...")
        self._save_images(x_test, y_test, self.test_dir)

        print("Dataset downloaded and saved successfully!")

    def apply_augmentation(self):
        """Applies each augmentation method and saves results in separate folders."""
        print("Applying augmentations...")
        augment_methods = [
            self._rotate_image,
            self._flip_image,
            self._crop_image,
            self._adjust_brightness,
            self._inject_noise,
            self._zoom_image,
            self._color_jitter,
            self._cutout_image,
        ]
        method_names = [
            "rotation",
            "flip",
            "crop",
            "brightness",
            "noise",
            "zoom",
            "color_jitter",
            "cutout",
        ]

        for method, name in zip(augment_methods, method_names):
            train_output_dir = os.path.join(self.augmented_dir, name)
            test_output_dir = os.path.join(self.augmented_dir, "test", name)

            # Apply augmentations to the training set
            self._augment_and_save(self.train_dir, train_output_dir, method)

            # TODO: should we do it on test files?
            # Apply augmentations to the test set
            self._augment_and_save(self.test_dir, test_output_dir, method)

        print("Augmentations applied and saved!")

    # Internal helper methods
    def _save_images(self, images, labels, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, (image, label) in tqdm(enumerate(zip(images, labels)), total=len(images)):
            label_dir = os.path.join(output_dir, str(label[0]))
            os.makedirs(label_dir, exist_ok=True)
            filepath = os.path.join(label_dir, f"{i}.png")
            Image.fromarray(image).save(filepath)

    def _augment_and_save(self, input_dir, output_dir, augment_method):
        os.makedirs(output_dir, exist_ok=True)
        for label_dir in os.listdir(input_dir):
            label_input_dir = os.path.join(input_dir, label_dir)
            label_output_dir = os.path.join(output_dir, label_dir)
            os.makedirs(label_output_dir, exist_ok=True)

            for image_file in tqdm(os.listdir(label_input_dir), desc=f"Processing {label_dir}"):
                image_path = os.path.join(label_input_dir, image_file)
                image = np.array(Image.open(image_path))
                for i in range(self.augmentations_per_image):
                    augmented_image = augment_method(image)
                    save_path = os.path.join(label_output_dir, f"{image_file.split('.')[0]}_{i}.png")
                    Image.fromarray(augmented_image).save(save_path)

    # Augmentation methods
    def _rotate_image(self, image):
        return tf.image.rot90(image, k=np.random.randint(1, 4)).numpy()

    def _flip_image(self, image):
        return tf.image.random_flip_left_right(image).numpy()

    def _crop_image(self, image):
        image = tf.image.resize_with_crop_or_pad(image, 36, 36)
        return tf.image.random_crop(image, size=[32, 32, 3]).numpy()

    def _adjust_brightness(self, image):
        return tf.image.random_brightness(image, max_delta=0.2).numpy()

    def _inject_noise(self, image):
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        return tf.clip_by_value(image + noise, 0, 255).numpy().astype(np.uint8)

    def _zoom_image(self, image):
        scales = np.random.uniform(1.0, 1.2)
        zoomed_image = tf.image.resize(image, [int(32 * scales), int(32 * scales)])
        zoomed_image = tf.image.resize_with_crop_or_pad(zoomed_image, 32, 32)
        return zoomed_image.numpy().astype(np.uint8)

    def _color_jitter(self, image):
        return tf.image.random_saturation(image, lower=0.8, upper=1.2).numpy()

    def _cutout_image(self, image):
        mask_size = np.random.randint(5, 10)
        h, w, _ = image.shape
        x = np.random.randint(0, h - mask_size)
        y = np.random.randint(0, w - mask_size)
        image[x:x + mask_size, y:y + mask_size] = 0
        return image

    def display_augmented_examples(self, num_examples=5):
        """Displays example augmented images."""
        augment_methods = [
            self._rotate_image,
            self._flip_image,
            self._crop_image,
            self._adjust_brightness,
            self._inject_noise,
            self._zoom_image,
            self._color_jitter,
            self._cutout_image,
        ]
        method_names = [
            "rotation",
            "flip",
            "crop",
            "brightness",
            "noise",
            "zoom",
            "color_jitter",
            "cutout",
        ]

        (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
        sample_images = x_train[:num_examples]

        fig, axes = plt.subplots(num_examples, len(augment_methods) + 1, figsize=(15, 15))
        for i, image in enumerate(sample_images):
            axes[i, 0].imshow(image)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")
            for j, (method, name) in enumerate(zip(augment_methods, method_names)):
                augmented_image = method(image)
                axes[i, j + 1].imshow(augmented_image)
                axes[i, j + 1].set_title(name)
                axes[i, j + 1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    processor = DatasetHandler(base_dir="./data", augmentations_per_image=5)
    processor.download_dataset()  # Step 1: Download CIFAR-10 dataset
    processor.apply_augmentation()  # Step 2: Apply data augmentations
    processor.display_augmented_examples(num_examples=5)