import PySimpleGUI as sg
from pathlib import Path
from PIL import Image, ImageTk
import os
import numpy as np
from keras.preprocessing import image
from Models.cnn_categorical import ConvolutionalNN
from Models.fused_model import FusedModel
from Assistants.datasetLoader import DatasetLoader

## PATH TO ALL THE MODELS ##
PATH_TO_MODELS = Path(os.path.dirname(os.path.realpath(__file__)))

CNN_FILTER_SAVE_LOC = Path(
    f"{PATH_TO_MODELS}/Models/Saved Models/cnn_filter.tf")

THREE_LAYER_CNN = Path(
    f"{PATH_TO_MODELS}/Models/Saved Models/animal10n.keras")


class WildLensGUI:
    def __init__(self):
        """
        Initialize WildLensGUI instance.
        """
        self.model = None
        self.window = None
        self.load_model("WildLens (WL18)")
        self.image_path = None
        self.ds = DatasetLoader(load_from_hub=False)

    def run(self):
        """
        Run the GUI.
        """
        self.start_gui()

    def start_gui(self):
        """
        Start the WildLens GUI.
        """
        sg.theme('DarkBlack1')

        layout = [
            [sg.Combo(['WildLens (WL18)', '8-Layer CNN', 'Fused Model'],
                      default_value="WildLens (WL18)",
                      key='-CHOOSE_MODEL-', size=(15, 3),
                      enable_events=True),
             sg.Button('Upload Image', key='-UPLOAD-', size=(15, 1)),
             sg.Button('Run', key='-RUN-', size=(15, 1)),
             ],
            [
                sg.Column([
                    [sg.Image(key='-IMAGE-', size=(256, 256), filename=self.image_path)],
                ], justification='center'),  # Centered first column

                sg.Column([
                    [sg.Text('', size=(50, 20), font=('Consolas', 14), key='-PREDICTIONS-')],
                ], justification='center'),  # Centered second column
            ],

            [sg.Button('Close', key='-BACK-', size=(10, 1))]
        ]

        self.window = sg.Window('WildLens', layout, return_keyboard_events=True,
                                size=(1000, 800), resizable=True, finalize=True,
                                element_justification='center', font=('Consolas', 18))

        self.main_loop()

    def main_loop(self):
        """
        Main event loop for WildLens GUI.
        """
        while True:
            event, values = self.window.read()

            if event in (sg.WINDOW_CLOSED, '-BACK-'):
                break
            elif event == '-UPLOAD-':
                self.load_image()
                self.update_display()
            elif event == '-CHOOSE_MODEL-':
                self.load_model(values['-CHOOSE_MODEL-'])
            elif event == '-RUN-':
                self.predict()

        self.window.close()

    def load_image(self):
        """
        Load an image selected by the user.
        """
        file_path = sg.popup_get_file('Upload Image', no_window=True)
        if file_path:
            self.image_path = file_path
            # Update the predictions_output element
            self.window['-PREDICTIONS-'].update("")

    def load_model(self, model_choice):
        """
        Load a pre-trained model based on user's choice.
        """
        if model_choice == 'WildLens (WL18)':
            self.model = ConvolutionalNN(str(CNN_FILTER_SAVE_LOC), input_shape=(64, 64, 3), output_shape=10)
        elif model_choice == '8-Layer CNN':
            self.model = ConvolutionalNN(str(THREE_LAYER_CNN), input_shape=(64, 64, 3), output_shape=10)
        elif model_choice == 'Fused Model':
            self.model = FusedModel(load_data=False)

        if self.model:
            self.model.load()
            try:
                # Update the predictions_output element
                self.window['-PREDICTIONS-'].update("")
            except:
                return

    def predict(self):
        """
        Make predictions using the loaded model on the selected image.
        """
        if self.model and self.image_path:
            img = image.load_img(self.image_path, target_size=(64, 64))

            # Convert the image to an array
            img_array = image.img_to_array(img)

            # Expand dimensions to match the expected input shape (add batch dimension)
            img_array = np.expand_dims(img_array, axis=0)

            # Preprocess the input (normalize pixel values)
            img_array = self.ds.standardize_data(img_array)
            predictions = self.model.predict(img_array)

        # Display the results
        self.display_results(predictions)

    def display_results(self, predictions):
        """
        Dispaly model's predicted output
        """
        # Convert log probabilities to percentages if needed
        if (1 - sum(predictions[0])) < -0.0001:
            predictions = np.exp(predictions) / np.sum(np.exp(predictions), axis=1)

        # Get the indices in descending order
        sorted_indices = np.argsort(predictions[0])[::-1]

        # Get the names of the animals based on the index
        animal_names = [self.get_animal_name(index) for index in sorted_indices]

        # Display the results in descending order
        result_text = f"Predictions:\n\n"
        for name, probability in zip(animal_names, predictions[0][sorted_indices]):
            result_text += f"{name}: {probability * 100:.2f}%\n"

        # Update the predictions_output element
        self.window['-PREDICTIONS-'].update(result_text)

    def update_display(self):
        """
        Update the GUI display with the loaded image.
        """
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((512, 512))
            photo_img = ImageTk.PhotoImage(image)
            self.window['-IMAGE-'].update(data=photo_img)

    def get_animal_name(self, label):
        """
        Get the name of the animal in the image from its label.
            lynx
            guinea pig
            jaguar
            cat
            hamster
            cheetah
            coyote
            chimpanzee
            wolf
            orangutan
        :param current_image:
        :return:
        """
        if label == 0:
            return 'Cat'
        elif label == 1:
            return 'Lynx'
        elif label == 2:
            return 'Wolf'
        elif label == 3:
            return 'Coyote'
        elif label == 4:
            return 'Cheetah'
        elif label == 5:
            return 'Jaguar'
        elif label == 6:
            return 'Chimpanzee'
        elif label == 7:
            return 'Orangutan'
        elif label == 8:
            return 'Hamster'
        elif label == 9:
            return 'Guinea Pig'
        else:
            return f"Unknown animal {label}"


def main():
    wl_gui = WildLensGUI()
    wl_gui.run()


if __name__ == '__main__':
    main()
