from .base import GAN, set_configuration, plot_history

from keras.layers import Input, Dense, Reshape, Flatten, BatchNormalization, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

import datetime
import numpy as np
import os

NUM_OF_EPOCHS = 30
BATCH_SIZE = 50
SAMPLE_INTERVAL = 200
DEBUG = 0


class DCGAN(GAN):

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
            g = Dropout(0.2)(g)
            return BatchNormalization(momentum=0.2)(g)

        noise = Input(shape=(self.latent_dim,))

        # My graphics card RAM amount wasn't enough to support more layers
        #g = g_dense_block(noise, int(self.img_size // 32) * int(self.img_size // 32) * self.gf * 8)
        g = g_dense_block(noise, int(self.img_size // 16) * int(self.img_size // 16) * self.gf * 8)
        g = g_dense_block(g, int(self.img_size // 8) * int(self.img_size // 8) * self.gf * 8)
        g = Reshape((int(self.img_size // 8), int(self.img_size // 8), self.gf * 8))(g)

        g = g_conv_block(g, self.gf * 8)
        g = g_conv_block(g, self.gf * 2)
        g = g_conv_block(g, self.gf)

        gen_img = Conv2D(self.channels, kernel_size=(1, 1), padding='same', activation='tanh')(g)

        return Model(noise, gen_img)


    def build_discriminator(self):

        def d_conv_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(layer_input)
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
        validity = Dense(1, activation='sigmoid')(d)

        return Model(img, validity)


    def train(self, epochs, batch_size, sample_interval):
        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        start_time = datetime.datetime.now()

        half_batch = int(batch_size / 2) or 1

        max_iter = int(self.n_data/batch_size)
        os.makedirs(f'.\\logs\\{self.time}', exist_ok=True)
        tensorboard = TensorBoard(f'.\\logs\\{self.time}')
        tensorboard.set_model(self.generator)

        os.makedirs(f'src\\models\\{self.time}', exist_ok=True)
        with open(f'src\\models\\{self.time}\\generator_architecture.json', 'w') as f:
            f.write(self.generator.to_json())
        print(f'\nbatch size : {batch_size} | num_data : {self.n_data} | max iteration : {max_iter} | time : {self.time} \n')

        for epoch in range(self.last_epoch+1, self.last_epoch+epochs+1):
            for i in range(max_iter):

                # Creating two batches with reference and fake images
                ref_imgs = self.data_loader.load_data(batch_size)

                # Select a random half batch of images
                idx = np.random.randint(0, ref_imgs.shape[0], half_batch)
                ref_imgs = ref_imgs[idx]
                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                d_loss_real = self.discriminator.train_on_batch(ref_imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generator wants to see its images as real
                g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

                # Logging messages and saving generated images
                tensorboard.on_epoch_end(i, named_logs(self.combined, [g_loss]))

                self.history.append({'D': d_loss[0], 'G': g_loss})

                if i % (sample_interval // 10) == 0:
                    elapsed_time = datetime.datetime.now() - start_time
                    print(f'epoch:{epoch} | iter : {i} / {max_iter} | time : {str(elapsed_time):10s} | '
                          f'g_loss : {g_loss:4.3f} | d_loss : {d_loss[0]:4.3f} | acc : {100*d_loss[1]:.3f}')

                if (i+1) % sample_interval == 0:
                    self.sample_images(epoch, i+1)

            self.last_epoch += 1

            # Saving models weights and loss history every 10 epochs
            if epoch % 10 == 0:
                self.save_model()

        # save the models weights and the loss history at the end
        self.save_model()


if __name__ == '__main__':

    set_configuration()
    optimizer = Adam(learning_rate=5e-5, beta_1=0.5)
    gan = DCGAN(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'],
                start_time=None,
                last_epoch=None)

    if DEBUG == 1:
        gan.train(epochs=2, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)
    else:
        gan.train(epochs=NUM_OF_EPOCHS, batch_size=BATCH_SIZE, sample_interval=SAMPLE_INTERVAL)

    plot_history(gan.history, gan.time)
