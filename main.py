from pathlib2 import Path

from dataset_creation.dataset_gather import download_pictures, verify_pictures, split_train_test
from Constants import LIST_OF_SPECIES, OUTPUT_DIR, NUM_OF_PICS
from cnn import cnn_model
from visualization import plot_accuracy

if __name__ == '__main__':

    # Conditions to access functionality of code
    download_pics = False
    verify_pics = False
    split = False
    run_cnn = True
    visualization=False

    if download_pics:
        for i in range(len(LIST_OF_SPECIES)):
            download_pictures(LIST_OF_SPECIES[i], NUM_OF_PICS)

    if verify_pics:
        verify_pictures(OUTPUT_DIR)

    if split:
        split_train_test(OUTPUT_DIR)

    if run_cnn:
        training_dataset = Path.cwd() / 'train'
        testing_dataset = Path.cwd() / 'test'
        model,acc, val_acc, loss, val_loss, epochs_range = cnn_model(training_dataset,testing_dataset)
        if visualization:
            plot_accuracy(acc,val_acc,loss,val_loss,epochs_range)