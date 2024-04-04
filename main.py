import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_cnn1(input_shape):
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    return model


def create_vgg16(input_shape):
    # Transfer learning: Used pre-trained VGG16 model
    vgg16_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in vgg16_base.layers:
        layer.trainable = False
    model = Sequential(
        [
            vgg16_base,
            # fine-tuning the top layer of pre-trained model
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


def create_cnn2(input_shape):
    model = Sequential(
        [
            Conv2D(64, (3, 3), activation="relu", input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(256, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(512, (3, 3), activation="relu"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(512, activation="relu"),
            Dropout(0.5),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )

    return model


"""This function is adopted from:
https://gist.github.com/RyanAkilos/3808c17f79e77c4117de35aa68447045?permalink_comment_id=4225817#gistcomment-4225817
"""


def plot_confusion_matrix(
    cm,
    classes,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
    model_index=None,
    figsize=(6, 6),
):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(f"figs/Model_{model_index + 1}_Confusion_matrix.png", dpi=150)


def train_and_evaluate_cnn_models(
    dataset_directory, test_directory, image_height, image_width, batch_sz, num_epochs
):
    # Create data generators
    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )
    val_data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    # Generating training data
    print("Generating training data!")
    train_gen = train_data_gen.flow_from_directory(
        dataset_directory,
        target_size=(image_height, image_width),
        batch_size=batch_sz,
        class_mode="binary",
        seed=42,
        subset="training",
    )
    # Generating validation data
    print("Generating validation data!")
    val_gen = val_data_gen.flow_from_directory(
        dataset_directory,
        target_size=(image_height, image_width),
        batch_size=batch_sz,
        class_mode="binary",
        seed=42,
        subset="validation",
    )

    test_data_gen = ImageDataGenerator(rescale=1.0 / 255)
    # Generating testing data
    print("Generating testing data!")
    test_gen = test_data_gen.flow_from_directory(
        directory=test_directory,
        target_size=(image_height, image_width),
        class_mode="binary",
        batch_size=batch_sz,
        seed=42,
        shuffle=False,
    )

    # Modelling
    input_shape = (image_height, image_width, 3)
    models = [
        create_cnn1(input_shape),
        create_vgg16(input_shape),
        create_cnn2(input_shape),
    ]

    for i, model in enumerate(models):
        print(f"Training and evaluating Model {i + 1}...")
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        training_history = model.fit(
            train_gen, epochs=num_epochs, validation_data=val_gen
        )

        validation_loss, validation_accuracy = model.evaluate(val_gen)
        print(f"Validation Loss for Model {i + 1}: {validation_loss}")
        print(f"Validation Accuracy for Model {i + 1}: {validation_accuracy}")

        # Make predictions and save it.
        predictions = np.round(model.predict(test_gen))
        print(f"Saving the predictions for Model {i + 1}...")
        np.save(f"outputs/predictions_model_{i + 1}.npy", predictions)

        # Plot train_loss v/s validation loss
        plt.plot(training_history.history["loss"], label="Train Loss")
        plt.plot(training_history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.title(f"Model {i + 1} Loss")
        plt.savefig(f"figs/Model_{i + 1}_Loss.png", dpi=150)

        # Plot train_accuracy v/s validation_accuracy
        plt.plot(training_history.history["accuracy"], label="Train Accuracy")
        plt.plot(training_history.history["val_accuracy"], label="Validation Accuracy")
        plt.legend()
        plt.title(f"Model {i + 1} Accuracy")
        plt.savefig(f"figs/Model_{i + 1}_Accuracy.png", dpi=150)

        # Plotting the confusion matrix
        cm = confusion_matrix(test_gen.classes, predictions)
        cm_plot_label = test_gen.class_indices
        plot_confusion_matrix(
            cm,
            cm_plot_label,
            title=f"Confusion Matrix of Model {i + 1}",
            model_index=i,
            figsize=(6, 6),
        )


if __name__ == "__main__":
    dataset_directory = "./data/train"
    test_directory = "./data/test"
    image_height, image_width = 64, 64
    batch_sz = 32
    num_epochs = 30
    train_and_evaluate_cnn_models(
        dataset_directory,
        test_directory,
        image_height,
        image_width,
        batch_sz,
        num_epochs,
    )
    print("Code executed successfully!")
