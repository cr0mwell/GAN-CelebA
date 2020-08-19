from __future__ import print_function, division

from keras import backend
from keras.constraints import Constraint
from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Dropout, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import datetime
from glob import glob
import numpy as np
import os
from time import localtime, strftime

NUM_OF_EPOCHS = 5
DEBUG = 1


def crop_face(import_path, export_path, target_img_size, casc_path='haarcascade_frontalface_default.xml'):
    '''
    Resizes the images from import_path to the target_size(tuple) and saves them to the export_path.
    Filters non-fullface images.
    '''

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
                image_crop = image[y: y+w, x : x+w, :]
                image_resize = cv2.resize(image_crop, target_img_size)
                cv2.imwrite(os.path.join(export_path, file), image_resize)
                print(file)
                count += 1
        else:
            pass

    print("total: %d / %d" % (count, len(files)))


# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)


class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):

        self.dataset_name = dataset_name
        self.img_res = img_res
        self.path = glob('src\\datasets\\%s\\%sx%s\\*.jpg' % (self.dataset_name, self.img_res[0], self.img_res[1]))
        self.no_of_files = len(self.path)


    def load_data(self, img_size, batch_size=1):
        '''
        Loads images and adds a noise to them.
        '''

        imgs = []
        #mean, var = 0, 0.1
        #sigma = var ** 0.5

        batch_images = np.random.choice(self.path, size=batch_size)

        for img_path in batch_images:
            # Creating noise
            #gauss = np.random.normal(mean, sigma, img_size)

            # Rescaling the image from [0.,255.] to [0.,1.]
            #img = np.array(img_to_array(load_img(img_path, target_size=img_size))).astype(np.float32) / 255.

            # Rescaling the image from [0.,255.] to [-1.,1.]
            img = (np.array(img_to_array(load_img(img_path, target_size=img_size))).astype(np.float32) - 127.5) / 127.5

            # Adding noise
            #imgs.append(img + gauss)
            imgs.append(img)

        return np.array(imgs)


    def get_img_number(self):
        return self.no_of_files


class DCGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.d_train_iterations = 5
        self.img_size = 64
        self.latent_dim = 100
        self.time = strftime('%m%d_%H%M', localtime())
        self.dataset_name = 'CelebA'

        optimizer = RMSprop(lr=5e-5)

        self.gf = 64  # filter size of generator's last layer
        self.df = 64  # filter size of discriminator's first layer

        # Configure data loader
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_size, self.img_size))
        self.n_data = self.data_loader.get_img_number()

        self.generator = self.build_generator()
        print("---------------------generator summary----------------------------")
        self.generator.summary()

        self.generator.compile(loss=wasserstein_loss, optimizer=optimizer)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        print("\n---------------------discriminator summary----------------------------")
        self.discriminator.summary()

        self.discriminator.compile(loss=wasserstein_loss, optimizer=optimizer)

        z = Input(shape=(self.latent_dim,))
        fake_img = self.generator(z)
        print(f'fake image {fake_img.shape}')

        # for the combined model, we only train generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        validity = self.discriminator(fake_img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, validity)
        print("\n---------------------combined summary----------------------------")
        self.combined.summary()
        self.combined.compile(loss=wasserstein_loss, optimizer=optimizer)


    def get_time(self):
        return self.time


    def build_generator(self):

        def g_dense_block(layer_input, shape):
            g = Dense(shape)(layer_input)
            g = LeakyReLU(0.2)(g)
            return BatchNormalization(momentum=0.2)(g)

        def g_conv_block(layer_input, filters, strides=(2, 2)):
            g = Conv2DTranspose(filters, kernel_size=(3, 3), strides=strides, padding='same')(layer_input)
            g = LeakyReLU(0.2)(g)
            g = Conv2D(self.gf, (3, 3), padding='same')(g)
            g = LeakyReLU(0.2)(g)
            return BatchNormalization(momentum=0.2)(g)

        noise = Input(shape=(self.latent_dim,))

        #g = g_dense_block(noise, int(self.img_size // 32) * int(self.img_size // 32) * self.gf * 8)
        #g = g_dense_block(noise, int(self.img_size // 16) * int(self.img_size // 16) * self.gf * 8)
        g = g_dense_block(noise, int(self.img_size // 8) * int(self.img_size // 8) * self.gf * 8)
        g = Reshape((int(self.img_size // 8), int(self.img_size // 8), self.gf * 8))(g)

        g = g_conv_block(g, self.gf * 8)
        g = g_conv_block(g, self.gf * 2)
        g = g_conv_block(g, self.gf)

        gen_img = Conv2D(self.channels, kernel_size=(1, 1), padding='same', activation='tanh')(g)

        return Model(noise, gen_img)


    def build_discriminator(self):

        const = ClipConstraint(0.01)

        def d_conv_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=const, padding='same')(layer_input)
            return LeakyReLU(0.2)(d)

        # Input img = generated image
        img = Input(shape=(self.img_size, self.img_size, self.channels))

        d = d_conv_block(img, self.df)
        d = d_conv_block(d, self.df * 2)
        d = d_conv_block(d, self.df * 4)

        d = Flatten()(d)
        d = Dense(1024)(d)
        d = LeakyReLU(0.2)(d)
        d = Dropout(0.2)(d)
        # Using linear activation (by default)
        validity = Dense(1)(d)

        return Model(img, validity)


    def train(self, epochs, batch_size, sample_interval):
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        start_time = datetime.datetime.now()

        half_batch = int(batch_size / 2) or 1

        history = []

        max_iter = int(self.n_data/batch_size)
        os.makedirs('.\\logs\\%s' % self.time, exist_ok=True)
        tensorboard = TensorBoard('.\\logs\\%s' % self.time)
        tensorboard.set_model(self.generator)

        os.makedirs('src\\models\\%s' % self.time, exist_ok=True)
        with open('src\\models\\%s\\%s_architecture.json' % (self.time, 'generator'), 'w') as f:
            f.write(self.generator.to_json())
        print("\nbatch size : %d | num_data : %d | max iteration : %d | time : %s \n" % (batch_size, self.n_data, max_iter, self.time))
        for epoch in range(1, epochs+1):
            for iter in range(max_iter):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                # For Wasserstein GAN we need to train the discriminator more than once per iteration
                d_losses, accuracies = [], []
                for _ in range(self.d_train_iterations):
                    # Creating two batches with reference and fake images
                    ref_imgs = self.data_loader.load_data((self.img_size, self.img_size, self.channels), batch_size)

                    # Select a random half batch of images
                    idx = np.random.randint(0, ref_imgs.shape[0], half_batch)
                    ref_imgs = ref_imgs[idx]
                    noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                    gen_imgs = self.generator.predict(noise)

                    d_loss_real = self.discriminator.train_on_batch(ref_imgs, -np.ones((half_batch, 1)))
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.ones((half_batch, 1)))
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    d_losses.append(d_loss)

                # taking the median from all collected losses
                d_loss = np.mean(d_losses)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Logging messages and saving generated images
                g_loss = self.combined.train_on_batch(noise, -np.ones((batch_size, 1)))
                tensorboard.on_epoch_end(iter, named_logs(self.combined, [g_loss]))

                history.append({"D": d_loss, "G": g_loss})

                if iter % (sample_interval // 10) == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print("epoch:%d | iter : %d / %d | time : %10s | g_loss : %4.3f | d_loss : %4.3f " %
                          (epoch, iter, max_iter, elapsed_time, g_loss, d_loss))

                if (iter+1) % sample_interval == 0:
                    self.sample_images(epoch, iter+1)

            # save weights after every epoch
            self.generator.save_weights('src\\models\\%s\\%s_epoch%d_weights.h5' % (self.time, 'generator', epoch))

        return history


    def sample_images(self, epoch, iter):
        os.makedirs('src\\samples\\%s' % self.time, exist_ok=True)
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        #gen_imgs = (1/2.5) * gen_imgs + 0.5

        # Save generated images and the high resolution originals
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col in range(c):
                axs[row, col].imshow(gen_imgs[cnt])
                axs[row, col].axis('off')
                cnt += 1
        fig.savefig("src\\samples\\%s\\e%d-i%d.png" % (self.time, epoch, iter), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    """
    # Resizing the dataset images
    crop_face('src\\datasets\\CelebA\\sources',
              'src\\datasets\\CelebA\\64x64', (64, 64),
              '..\\venv\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    """
    backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    gan = DCGAN()
    if DEBUG == 1:
        history = gan.train(epochs=2, batch_size=50, sample_interval=200)

    else:
        history = gan.train(epochs=NUM_OF_EPOCHS, batch_size=50, sample_interval=200)

    hist = pd.DataFrame(history)
    plt.figure(figsize=(20, 5))
    for colnm in hist.columns:
        plt.plot(hist[colnm], label=colnm)
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig("src\\samples\\%s\\Loss history.png" % gan.get_time(), bbox_inches='tight', pad_inches=0)
    plt.close()
