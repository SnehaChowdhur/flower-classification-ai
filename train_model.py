import tensorflow as tf
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split


# ==============================
# 1. BASIC SETTINGS
# ==============================

DATASET_PATH = "DATASET"  # Main folder containing all flower classes

IMG_SIZE = (224, 224)  # Resize every image to the same size

BATCH_SIZE = 32  # Number of images processed together

EPOCHS = 5  # Number of times the model sees the training dataset

SEED = 42  # Keeps the dataset split reproducible


# ==============================
# 2. COLLECT IMAGE PATHS
# ==============================

image_paths = []  # Stores the path of every flower image

labels = []  # Stores the flower name belonging to each image

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
)  # Image file formats that we accept


# Go through every flower folder inside DATASET
for flower_name in os.listdir(DATASET_PATH):

    flower_folder = os.path.join(
        DATASET_PATH,
        flower_name
    )  # Create the path of the flower folder

    if not os.path.isdir(flower_folder):
        continue  # Ignore anything that is not a folder

    # Search inside the flower folder and its subfolders
    # This handles both Bulk and Single folders
    for root, directories, filenames in os.walk(flower_folder):

        for file in filenames:

            if file.endswith(valid_extensions):
                # Check whether the file is an image

                image_path = os.path.join(
                    root,
                    file
                )  # Create the complete image path

                image_paths.append(
                    image_path
                )  # Store the image path

                labels.append(
                    flower_name
                )  # Store the flower name as its label


# ==============================
# 3. CREATE DATAFRAME
# ==============================

data = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})  # Create a table containing paths and labels


print("\nTotal images:", len(data))

print("\nImages per class:")

print(
    data["label"].value_counts()
)  # Count how many images belong to each flower


# ==============================
# 4. CREATE CLASS NUMBERS
# ==============================

class_names = sorted(
    data["label"].unique()
)  # Get all different flower names


num_classes = len(
    class_names
)  # Count the number of flower classes


print("\nClasses:")

for index, class_name in enumerate(class_names):

    print(
        index,
        "->",
        class_name
    )  # Display the number assigned to each flower


print(
    "\nNumber of classes:",
    num_classes
)


# Create a dictionary:
# flower name -> number
class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}


data["label_index"] = data["label"].map(
    class_to_index
)  # Convert flower names into numbers


# ==============================
# 5. SPLIT THE DATASET
# ==============================

train_data, temporary_data = train_test_split(
    data,
    test_size=0.30,
    stratify=data["label_index"],
    random_state=SEED
)  # 70% training, 30% temporary


validation_data, test_data = train_test_split(
    temporary_data,
    test_size=0.50,
    stratify=temporary_data["label_index"],
    random_state=SEED
)  # Split remaining data equally


print("\nDataset split:")

print(
    "Training:",
    len(train_data)
)

print(
    "Validation:",
    len(validation_data)
)

print(
    "Testing:",
    len(test_data)
)


# ==============================
# 6. IMAGE PREPROCESSING
# ==============================

def load_image(image_path, label):

    image = tf.io.read_file(
        image_path
    )  # Read the image file

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )  # Decode JPG/PNG images

    image = tf.image.resize(
        image,
        IMG_SIZE
    )  # Resize image to 224x224

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0  # Convert pixel values from 0-255 to 0-1

    return image, label  # Return processed image and its label


# ==============================
# 7. GET PATHS AND LABELS
# ==============================

train_paths = train_data[
    "image_path"
].values  # Training image paths

train_labels = train_data[
    "label_index"
].values  # Training labels


validation_paths = validation_data[
    "image_path"
].values  # Validation image paths

validation_labels = validation_data[
    "label_index"
].values  # Validation labels


test_paths = test_data[
    "image_path"
].values  # Testing image paths

test_labels = test_data[
    "label_index"
].values  # Testing labels


# ==============================
# 8. CREATE TENSORFLOW DATASETS
# ==============================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)  # Create training dataset


validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)  # Create validation dataset


test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)  # Create testing dataset


# Apply image preprocessing
train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ==============================
# 9. DATA AUGMENTATION
# ==============================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),  # Randomly flip images

    tf.keras.layers.RandomRotation(
        0.1
    ),  # Randomly rotate images

    tf.keras.layers.RandomZoom(
        0.1
    )  # Randomly zoom images

])


# ==============================
# 10. MOBILE NET V2
# ==============================

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(224, 224, 3),

    include_top=False,

    weights="imagenet"

)  # Use MobileNetV2 pretrained on ImageNet


base_model.trainable = False
# Freeze pretrained layers during initial training


# ==============================
# 11. BUILD OUR CLASSIFIER
# ==============================

model = tf.keras.Sequential([

    data_augmentation,

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),
    # Convert feature maps into a single feature vector

    tf.keras.layers.Dropout(0.2),
    # Reduce overfitting

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
    # Output one probability for every flower class

])


# ==============================
# 12. COMPILE MODEL
# ==============================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),  # Controls how the model learns

    loss="sparse_categorical_crossentropy",
    # Measures classification error

    metrics=["accuracy"]
    # Track prediction accuracy

)


model.summary()


# ==============================
# 13. TRAIN THE MODEL
# ==============================

history = model.fit(

    train_dataset
    .shuffle(1000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE),

    validation_data=validation_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE),

    epochs=EPOCHS

)


# ==============================
# 14. TEST THE MODEL
# ==============================

test_loss, test_accuracy = model.evaluate(

    test_dataset.batch(BATCH_SIZE)

)


print(
    "\nTest Loss:",
    test_loss
)

print(
    "Test Accuracy:",
    test_accuracy
)


# ==============================
# 15. SAVE THE MODEL
# ==============================

model.save(
    "flower_classifier.keras"
)  # Save trained model


np.save(
    "class_names.npy",
    np.array(class_names)
)  # Save flower names for prediction later


print(
    "\nModel saved as flower_classifier.keras"
)

print(
    "Class names saved as class_names.npy"
)
'''
🚀 Your overall project pipeline

You'll eventually have:

13 Flower Classes
       ↓
Pandas DataFrame
       ↓
Train / Validation / Test
       ↓
Image Preprocessing
       ↓
Data Augmentation
       ↓
MobileNetV2
       ↓
Training
       ↓
Evaluation
       ↓
flower_classifier.keras
       ↓
FastAPI
       ↓
Flutter / Website
       ↓
📷 User takes flower photo
       ↓
API receives image
       ↓
Model predicts flower
       ↓
🌸 "This is Chandramallika"
'''
'''
FOR WEBSITE WORKFLOW:

                 🌸 FLOWER CLASSIFICATION
                         │
                         ▼
                 7,989 images
                         │
                         ▼
                    13 classes
                         │
                         ▼
                    MobileNetV2
                         │
                         ▼
                 95.25% Test Accuracy
                         │
                         ▼
             flower_classifier.keras
                         │
                         ▼
                     FastAPI
                         │
                         ▼
                    🌐 Website
                         │
                         ▼
               📷 Upload flower image
                         │
                         ▼
                  Model prediction
                         │
                         ▼
              🌸 Flower name + confidence'''