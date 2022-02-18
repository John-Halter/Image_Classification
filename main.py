from matplotlib import pyplot as plt
from pathlib2 import Path

from dataset_creation.dataset_gather import download_pictures, verify_pictures, split_train_test
from Constants import LIST_OF_SPECIES, OUTPUT_DIR, NUM_OF_PICS, ITERATIONS
from cnn import cnn_model
from visualization import plot_accuracy, plot_model_outline, plot_layers, plot_feature_map, plot_multi_accuracy

if __name__ == '__main__':
    training_dataset = Path.cwd() / 'train'
    testing_dataset = Path.cwd() / 'test'

    # Conditions to access functionality of code
    download_pics = False
    verify_pics = False
    split = False
    run_cnn = False
    run_cnn_multi = True
    model_outline = False
    layers = False
    feature = False

    if download_pics:
        for i in range(len(LIST_OF_SPECIES)):
            download_pictures(LIST_OF_SPECIES[i], NUM_OF_PICS)

    if verify_pics:
        verify_pictures(OUTPUT_DIR)

    if split:
        split_train_test(OUTPUT_DIR)

    if run_cnn:
        model, acc, val_acc, loss, val_loss, epochs_range = cnn_model(training_dataset, testing_dataset)
        plot_accuracy(acc, val_acc, loss, val_loss, epochs_range)
        if model_outline:
            plot_model_outline(model)
        if layers:
            plot_layers(model)
        if feature:
            plot_feature_map(model)
    if run_cnn_multi:
        acc_ls, val_acc_ls,loss_ls, val_loss_ls, epochs_range_ls = [], [], [], [], []
        for i in range(1,ITERATIONS + 1):
            model, acc, val_acc, loss, val_loss, epochs_range = cnn_model(training_dataset, testing_dataset)
            acc_ls.append(acc)
            val_acc_ls.append(val_acc)
            loss_ls.append(loss)
            val_loss_ls.append(val_loss)
            epochs_range_ls.append(epochs_range)
        plot_multi_accuracy(acc_ls,val_acc_ls,loss_ls,val_loss_ls,epochs_range_ls)

plt.show()
