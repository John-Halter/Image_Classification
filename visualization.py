import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

def plot_accuracy(acc,val_acc,loss,val_loss,epochs_range):
    fig, ax = plt.subplots(2, figsize=(13, 8))
    ax[0].plot(epochs_range, acc, label='Training Accuracy')
    ax[0].plot(epochs_range, val_acc, label='Validation Accuracy')
    ax[0].set_ylabel("Accuracy Percentage")
    ax[0].legend(loc='lower right')
    ax[0].set_title('Training and Validation Accuracy')

    ax[1].plot(epochs_range, loss, label='Training Loss')
    ax[1].plot(epochs_range, val_loss, label='Validation Loss')
    ax[1].set_xlabel("Number of Epochs")
    ax[1].legend(loc='upper right')
    ax[1].set_title('Training and Validation Loss')




 # # dot_img_file = str(Path.cwd() / 'model1.png')
    # tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    # import tensorflow as tf
    # from keras.models import Sequential
    # from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
    # from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
    # img_path = str(Path.cwd() / 'train' / 'colorado hairstreak' / '000001.jpg')  # dog
    # # Define a new Model, Input= image
    # # Output= intermediate representations for all layers in the
    # # previous model after the first.
    # successive_outputs = [layer.output for layer in model.layers[1:]]
    # # visualization_model = Model(img_input, successive_outputs)
    # visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)
    # # Load the input image
    # img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    # # Convert ht image to Array of dimension (150,150,3)
    # x = img_to_array(img)
    # x = x.reshape((1,) + x.shape)
    # # Rescale by 1/255
    # x /= 255.0
    # # Let's run input image through our vislauization network
    # # to obtain all intermediate representations for the image.
    # successive_feature_maps = visualization_model.predict(x)
    # # Retrieve are the names of the layers, so can have them as part of our plot
    # layer_names = [layer.name for layer in model.layers]
    # for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    #     print(feature_map.shape)
    #     if len(feature_map.shape) == 4:
    #
    #         # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers
    #
    #         n_features = feature_map.shape[-1]  # number of features in the feature map
    #         size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)
    #
    #         # We will tile our images in this matrix
    #         display_grid = np.zeros((size, size * n_features))
    #
    #         # Postprocess the feature to be visually palatable
    #         for i in range(n_features):
    #             x = feature_map[0, :, :, i]
    #             x -= x.mean()
    #             x /= x.std()
    #             x *= 64
    #             x += 128
    #             x = np.clip(x, 0, 255).astype('uint8')
    #             # Tile each filter into a horizontal grid
    #             display_grid[:, i * size: (i + 1) * size] = x
    #         # Display the grid
    #         scale = 20. / n_features
    #         plt.figure(figsize=(scale * n_features, scale))
    #         plt.title(layer_name)
    #         plt.grid(False)
    #         plt.imshow(display_grid, aspect='auto', cmap='viridis')

    # # Iterate thru all the layers of the model
    # for layer in model.layers:
    #     if 'conv' in layer.name:
    #         weights, bias = layer.get_weights()
    #
    #         # normalize filter values between  0 and 1 for visualization
    #         f_min, f_max = weights.min(), weights.max()
    #         filters = (weights - f_min) / (f_max - f_min)
    #         filter_cnt = 1
    #
    #         # plotting all the filters
    #         for i in range(filters.shape[3]):
    #             # get the filters
    #             filt = filters[:, :, :, i]
    #             # plotting each of the channel, color image RGB channels
    #             for j in range(filters.shape[0]):
    #                 ax = plt.subplot(filters.shape[3], filters.shape[0], filter_cnt)
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
    #                 plt.imshow(filt[:, :, j])
    #                 filter_cnt += 1
    #         plt.show()











plt.show()
