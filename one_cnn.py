"""
One CNN model for the animal10N dataset.
"""
from sklearn.model_selection import train_test_split

from Assistants.datasetLoader import DatasetLoader
from Models.cnn_categorical import ConvolutionalNN
from Models.cnn_binary import CnnBinary
from pathlib import Path
import tensorflow as tf
from Assistants.harry_plotter import *
import os

CNN_FILTER_SAVE_LOC = Path(
    f"{Path(os.path.dirname(os.path.realpath(__file__)))}/Models/Saved Models/cnn_filter.tf")

COMPUTATIONAL_LIMITS = 50000

def main(
        train=False,
):
    ds = DatasetLoader()
    cnn = ConvolutionalNN(str(CNN_FILTER_SAVE_LOC), input_shape=(64, 64, 3), output_shape=10)

    # Preprocess the data.
    train_data, train_labels, test_data, test_labels = ds.process_data(size=COMPUTATIONAL_LIMITS)

    graphing_label = [
        'Cat', 'Lynx', 'Wolf', 'Coyote', 'Cheetah', 'Jaguar', 'Chimpanzee', 'Orangutan', 'Hamster',
        'Guinea Pig'
    ]

    cm_title = 'Animal10N Confusion Matrix'

    # Generate a validation set
    run_train_data, run_train_labels, valid_data, valid_labels = ds.generate_validation_set(
        train_data, train_labels
    )
    # One hot encode the labels
    run_train_labels = ds.one_hot_vectorize(run_train_labels)
    valid_labels = ds.one_hot_vectorize(valid_labels)
    # Train the model
    if train:
        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0,
                                             mode='min'),
            tf.keras.callbacks.ModelCheckpoint(str(CNN_FILTER_SAVE_LOC), save_best_only=True,
                                               monitor='val_loss', mode='min'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                                                 verbose=1, mode='min', min_lr=1e-7, cooldown=2)
        ]
        history = cnn.train(
            run_train_data,
            run_train_labels,
            epochs=100,
            batch_size=512,
            validation_data=(valid_data, valid_labels),
            callbacks=my_callbacks
        )
        plot_history(history)
        print('Training complete.')

    else:
        # Load the model
        cnn.load()
        print('Model loaded.')
    cnn.summary()

    test_labels = ds.one_hot_vectorize(test_labels)

    # Evaluate the model
    results = cnn.evaluate(test_data, test_labels)
    plot_evaluation(results)
    print('Accuracy:', results[1])
    print('Recall:', results[2])
    print('Precision:', results[3])
    print('F1:', results[4])

    # Create a confusion matrix
    cm = cnn.confusion_matrix(test_data, test_labels)
    plot_confusion_matrix(cm, graphing_label, title=cm_title)


if __name__ == '__main__':
    main(train=False)
