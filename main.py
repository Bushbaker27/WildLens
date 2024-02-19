from Assistants.datasetLoader import DatasetLoader
from Models.cnn_categorical import ConvolutionalNN
from Models.cnn_binary import CnnBinary
from pathlib import Path
import tensorflow as tf
from Assistants.harry_plotter import *
import os

CNN_FILTER_SAVE_LOC = Path(
    f"{Path(os.path.dirname(os.path.realpath(__file__)))}/Models/Saved Models/cnn_filter.keras")
ANIMAL_CATEGORICAL_SAVE_LOC = Path(
    f"{Path(os.path.dirname(os.path.realpath(__file__)))}/Models/Saved Models/")


def main(
        train=False,
):
    ds = DatasetLoader()
    # Animal disjoint models.
    cnn_batch_1 = ConvolutionalNN(str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_batch_1.tf')
    cnn_batch_2 = ConvolutionalNN(str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_batch_2.tf')

    # Animal joint models.
    cnn_cat_lynx = CnnBinary(str(ANIMAL_CATEGORICAL_SAVE_LOC) +
                             '/cnn_cat_lynx.tf')
    cnn_wolf_coyote = CnnBinary(str(ANIMAL_CATEGORICAL_SAVE_LOC) +
                                '/cnn_wolf_coyote.tf')
    cnn_cheetah_jaguar = CnnBinary(str(ANIMAL_CATEGORICAL_SAVE_LOC) +
                                   '/cnn_cheetah_jaguar.tf')
    cnn_chimp_orangutan = CnnBinary(str(ANIMAL_CATEGORICAL_SAVE_LOC) +
                                    '/cnn_chimp_orangutan.tf')
    cnn_hamster_guinea_pig = CnnBinary(str(ANIMAL_CATEGORICAL_SAVE_LOC) +
                                       '/cnn_hamster_guinea_pig.tf')

    filenames = [
        str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_batch_1.tf',
        str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_batch_2.tf',
        str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_cat_lynx.tf',
        str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_wolf_coyote.tf',
        str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_cheetah_jaguar.tf',
        str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_chimp_orangutan.tf',
        str(ANIMAL_CATEGORICAL_SAVE_LOC) + '/cnn_hamster_guinea_pig.tf'
    ]

    # Preprocess the data.
    train_data, train_labels, test_data, test_labels = ds.process_data()
    # Split the data into batches between disjoint sets.
    train_data_batch_1, train_labels_batch_1, train_data_batch_2, train_labels_batch_2 = \
        ds.split_data_by_batch(train_data, train_labels)
    test_data_batch_1, test_labels_batch_1, test_data_batch_2, test_labels_batch_2 = \
        ds.split_data_by_batch(test_data, test_labels)
    # Split the data into batches between joint sets.
    animal_sets = ds.split_data_by_similar_animal(train_data, train_labels)
    test_animal_sets = ds.split_data_by_similar_animal(test_data, test_labels)

    training_data = [train_data_batch_1, train_data_batch_2] + [set_a[0] for set_a in animal_sets]
    training_labels = [train_labels_batch_1, train_labels_batch_2] + [set_a[1] for set_a in
                                                                      animal_sets]

    testing_data = [test_data_batch_1, test_data_batch_2] + [set_a[0] for set_a in test_animal_sets]
    testing_labels = [test_labels_batch_1, test_labels_batch_2] + [set_a[1] for set_a in
                                                                   test_animal_sets]
    cnns = [cnn_batch_1, cnn_batch_2, cnn_cat_lynx, cnn_wolf_coyote, cnn_cheetah_jaguar,
            cnn_chimp_orangutan, cnn_hamster_guinea_pig]

    graphing_labels = [
        ['Cat', 'Wolf', 'Cheetah', 'Chimpanzee', 'Hamster'],
        ['Lynx', 'Coyote', 'Jaguar', 'Orangutan', 'Guinea Pig'],
        ['Cat', 'Lynx'],
        ['Wolf', 'Coyote'],
        ['Cheetah', 'Jaguar'],
        ['Chimpanzee', 'Orangutan'],
        ['Hamster', 'Guinea Pig']
    ]

    cm_titles = [
        'Batch 1',
        'Batch 2',
        'Cat vs Lynx',
        'Wolf vs Coyote',
        'Cheetah vs Jaguar',
        'Chimpanzee vs Orangutan',
        'Hamster vs Guinea Pig'
    ]
    epochs_per_cnn = [50, 50, 20, 20, 20, 20, 20]

    for i, cnn in enumerate(cnns):
        print('Training model', i)
        run_train_data, run_train_labels = training_data[i], training_labels[i]
        run_test_data, run_test_labels = testing_data[i], testing_labels[i]
        # Generate a validation set
        run_train_data, run_train_labels, valid_data, valid_labels = ds.generate_validation_set(
            run_train_data, run_train_labels)
        # Train the model
        if train:
            my_callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0,
                                                 mode='min'),
                tf.keras.callbacks.ModelCheckpoint(filenames[i], save_best_only=True,
                                                   monitor='val_loss', mode='min'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
                                                     verbose=1, mode='min', min_lr=1e-7, cooldown=2)
            ]
            history = cnn.train(
                run_train_data,
                run_train_labels,
                epochs=epochs_per_cnn[i],
                batch_size=512,
                validation_data=(valid_data, valid_labels),
                callbacks=my_callbacks
            )
            plot_history(history)


        else:
            # Load the model
            cnn.load()
        print('Training complete.')

        # Evaluate the model
        results = cnn.evaluate(run_test_data, run_test_labels)
        plot_evaluation(results)

        # Create a confusion matrix
        cm = cnn.confusion_matrix(run_test_data, run_test_labels)
        plot_confusion_matrix(cm, graphing_labels[i], title=cm_titles[i])


if __name__ == '__main__':
    main(train=True)
