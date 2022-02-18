import matplotlib
import tensorflow as tf
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

from Constants import IMG_WIDTH, IMG_HEIGHT, EPOCHS

matplotlib.use('TkAgg')


# TODO: Fix error message with gpu not being used for tensorflow
def cnn_model(training_dataset, testing_dataset):

    train_gen = _data_generator()
    test_gen = _data_generator()

    training_datagen = _new_data_generated(training_dataset,train_gen)
    testing_datagen = _new_data_generated(testing_dataset,test_gen)

    class_names = training_datagen.class_indices  # class_indices
    num_classes = len(class_names)

    model = Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    # Training done here
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(
        training_datagen,
        validation_data=testing_datagen,
        epochs=EPOCHS,
        verbose=2
    )

    acc = history.history['accuracy']
    acc = [val * 100 for val in acc]
    val_acc = history.history['val_accuracy']
    val_acc = [val * 100 for val in val_acc]

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)

    return model,acc, val_acc, loss, val_loss, epochs_range


def _data_generator():
    data_augmented = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True)
    return data_augmented


def _new_data_generated(dataset,datagen):
    new_data = datagen.flow_from_directory(
        dataset,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode="categorical")
    return new_data