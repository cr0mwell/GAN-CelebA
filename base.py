from keras.layers import Input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from glob import glob
import json
import numpy as np
import os
from time import localtime, strftime


def crop_face(import_path, export_path, target_img_size, casc_path='haarcascade_frontalface_default.xml'):
    """
    Resizes the images from import_path to the target_size(tuple) and saves them to the export_path.
    Filters non-fullface images.
    """

    face_classifier = cv2.CascadeClassifier(casc_path)
    files = os.listdir(import_path)
    count = 0

    for file in files:
        if file[-4:] == '.jpg':
            image = cv2.imread(os.path.join(import_path, file))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 5, 5)

            if len(faces) != 1:
                pass
            else:
                x, y, w, h = faces[0]
                image_crop = image[y:y+w, x:x+w, :]
                image_resize = cv2.resize(image_crop, target_img_size)
                cv2.imwrite(os.path.join(export_path, file), image_resize)
                print(file)
                count += 1
        else:
            pass

    print(f'total: {count} / {len(files)}')


def plot_history(history, time_stamp):
    """
    Plotts and saves the loss history graph.
    """

    hist = pd.DataFrame(history)
    plt.figure(figsize=(20, 5))
    for column in hist.columns:
        plt.plot(hist[column], label=column)
    plt.legend()
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.savefig(f'src\\samples\\{time_stamp}\\Loss history.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def set_configuration():
    """
    Sets Keras session configuration options
    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


class DataLoader:
    def __init__(self, dataset_name, img_res=128, img_channels=3):

        self.dataset_name = dataset_name
        self.img_res = img_res
        self.img_channels = img_channels
        self.path = glob(f'src\\datasets\\{self.dataset_name}\\{self.img_res}x{self.img_res}\\*.jpg')
        self.no_of_files = len(self.path)

    def load_data(self, batch_size=1):
        """
        Loads images and adds a noise to them.
        """

        imgs = []
        mean, var = 0, 0.001
        sigma = var ** 0.5

        batch_images = np.random.choice(self.path, size=batch_size)

        for img_path in batch_images:
            # Creating noise
            gauss = np.random.normal(mean, sigma, (self.img_res, self.img_res, self.img_channels))

            # Rescaling the image from [0.,255.] to [0.,1.] (if generator uses sigmoid activation)
            #img = np.array(img_to_array(load_img(img_path, target_size=img_size))).astype(np.float32) / 255.

            # Rescaling the image from [0.,255.] to [-1.,1.] (if generator uses tanh activation)
            img = (np.array(img_to_array(load_img(img_path, target_size=(self.img_res, self.img_res)))).astype(np.float32) - 127.5) / 127.5

            # Adding noise
            imgs.append(img + gauss)
            imgs.append(img)

        return np.array(imgs)


    def get_img_number(self):
        return self.no_of_files


class GAN:
    def __init__(self, loss, optimizer, metrics, start_time=None, last_epoch=None, dataset='CelebA',
                 img_size=64, channels=3, latent_dim=100, generator_filter=64, discriminator_filter=64):
        """
        Base class for GAN models.

        :param loss: String (name of objective function), objective function or tf.keras.losses.Loss instance.
                     See tf.keras.losses.
        :param optimizer: String (name of optimizer) or optimizer instance. See tf.keras.optimizers.
        :param metrics: List of metrics (or None) to be evaluated by the model during training and testing.
                        Each of this can be a string (name of a built-in function), function
                        or a tf.keras.metrics.Metric instance. See tf.keras.metrics.
        :param dataset: String. Name of the dataset that will be used to train the models.
        :param img_size: Int. size of the images from the train dataset
        :param channels: Int. Number of image channels from the train dataset
        :param latent_dim: Int. Latent dimension size
        :param generator_filter: Int. Filter size of generator's last layer
        :param discriminator_filter: Int. Filter size of discriminator's first layer

        Parameters used to continue training of previously trained models:
        :param start_time: String. Time stamp used by previously trained models.
        :param last_epoch: Int. Last epoch when the model weights were saved to files.

        """
        self.time = strftime('%m%d_%H%M', localtime()) if start_time is None else start_time
        self.last_epoch = 0 if last_epoch is None else last_epoch
        self.history = []

        # Hyperparameters
        self.channels = channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.dataset_name = dataset
        self.gf = generator_filter
        self.df = discriminator_filter

        # Configure data loader
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=self.img_size, )
        self.n_data = self.data_loader.get_img_number()

        self.generator = self.build_generator()
        print('---------------------generator summary----------------------------')
        self.generator.summary()

        self.generator.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=metrics)

        # If 'last_epoch' was given we should load the generator weights
        # as we do not intend to start the training from the beginning
        if last_epoch is not None:
            generator_weights_file = f'src\\models\\{self.time}\\generator_epoch{self.last_epoch}_weights.h5'
            if os.path.exists(generator_weights_file):
                self.generator.load_weights(generator_weights_file)
                print("Successfully loaded generator's weights")
            else:
                print(f'Failed to load generator\'s weights: no such file "{generator_weights_file}"')

            # Loading loss history
            history_file = f'src\\models\\{self.time}\\history.json'
            if os.path.exists(history_file):
                with open(history_file) as f:
                    self.history = json.load(f)
                    print("Successfully loaded history")
            else:
                print(f'Failed to load loss history: no such file "{history_file}"')

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        print('\n---------------------discriminator summary----------------------------')
        self.discriminator.summary()

        self.discriminator.compile(loss=loss,
                                   optimizer=optimizer,
                                   metrics=metrics)

        # If 'last_epoch' was given we should load the discriminator weights
        # as we do not intend to start the training from the beginning
        if last_epoch is not None:
            discriminator_weights_file = f'src\\models\\{self.time}\\discriminator_epoch{self.last_epoch}_weights.h5'
            if os.path.exists(discriminator_weights_file):
                self.discriminator.load_weights(discriminator_weights_file)
                print("Successfully loaded discriminator's weights")
            else:
                print(f'Failed to load discriminator\'s weights: no such file "{discriminator_weights_file}"')

        z = Input(shape=(self.latent_dim,))
        fake_img = self.generator(z)

        # for the combined model, we only train generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        validity = self.discriminator(fake_img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, validity)
        print('\n---------------------combined summary----------------------------')
        self.combined.summary()
        self.combined.compile(loss=loss,
                              optimizer=optimizer)


    def build_generator(self):
        raise NotImplementedError('This method should be implemented in the subclass')


    def build_discriminator(self):
        raise NotImplementedError('This method should be implemented in the subclass')


    def save_model(self):
        """
        Saves the models weights and loss history to files.
        """
        self.generator.save_weights(f'src\\models\\{self.time}\\generator_epoch{self.last_epoch}_weights.h5')
        self.discriminator.save_weights(f'src\\models\\{self.time}\\discriminator_epoch{self.last_epoch}_weights.h5')

        with open(f'src\\models\\{self.time}\\history.json', 'w') as f:
            json.dump(self.history, f)


    def train(self, epochs, batch_size, sample_interval):
        raise NotImplementedError('This method should be implemented in the subclass')


    def sample_images(self, epoch, iteration):
        os.makedirs(f'src\\samples\\{self.time}', exist_ok=True)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (1/2.5) * gen_imgs + 0.5

        # Save generated images and the high resolution originals
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col in range(c):
                axs[row, col].imshow(gen_imgs[cnt])
                axs[row, col].axis('off')
                cnt += 1
        fig.savefig(f'src\\samples\\{self.time}\\e{epoch}-i{iteration}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':

    # Resizing the dataset images
    crop_face('src\\datasets\\CelebA\\sources',
              'src\\datasets\\CelebA\\64x64', (64, 64),
              '..\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
