import tensorflow as tf

def create_training_data(height,width, dir,dataset_type):
    dataset_train = tf.keras.utils.image_dataset_from_directory( dir, validation_split=0.2, subset=dataset_type,
        seed=123, image_size=(height, width))
    return dataset_train