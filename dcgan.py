from base import GAN, get_datasets, step
from constants import (SEP, BATCH_SIZE, EPOCHS, ADAM_MOMENTUM, MODELS_DIR)

from argparse import ArgumentParser, BooleanOptionalAction

import tensorflow as tf
from tensorflow.keras.layers import (Dense, Reshape, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D,
                                     Conv2DTranspose, Conv2D, ReLU, LeakyReLU, Rescaling, Resizing, Lambda, Activation)
from tensorflow.keras import Input, Model


class DCGAN(GAN):

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(generator_optimizer, discriminator_optimizer, **kwargs)

        self.generator_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.real_accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.augmentation_probability_tracker,
            self.real_accuracy,
            self.kid,
        ]

    def adversarial_loss(self, real_logits, generated_logits):
        # this is usually called the non-saturating GAN loss

        real_labels = tf.ones(shape=(self.batch_size, 1))
        generated_labels = tf.zeros(shape=(self.batch_size, 1))

        # the generator tries to produce images that the discriminator considers as real
        generator_loss = tf.keras.losses.binary_crossentropy(real_labels, generated_logits, from_logits=True)
        # the discriminator tries to determine if images are real or generated
        discriminator_loss = tf.keras.losses.binary_crossentropy(
            tf.concat([real_labels, generated_labels], axis=0),
            tf.concat([real_logits, generated_logits], axis=0),
            from_logits=True
            )

        return tf.reduce_mean(generator_loss), tf.reduce_mean(discriminator_loss)

    def train_step(self, real_images):
        real_images = self.augmenter(real_images, self.batch_size, training=True)

        # Training the discriminator 2 steps and generator 1 step
        for _ in range(self.d_steps):
            # use persistent gradient tape because gradients will be calculated twice
            with tf.GradientTape(persistent=True) as tape:

                generated_images = self.generate(self.batch_size, training=True)
                # gradient is calculated through the image augmentation
                generated_images = self.augmenter(generated_images, self.batch_size, training=True)

                # separate forward passes for the real and generated images, meaning
                # that batch normalization is applied separately
                real_logits = self.discriminator(real_images, training=True)
                generated_logits = self.discriminator(generated_images, training=True)

                generator_loss, discriminator_loss = self.adversarial_loss(
                    real_logits, generated_logits
                )

            # calculate gradients and update weights for generator
            discriminator_gradients = tape.gradient(
                discriminator_loss, self.discriminator.trainable_weights
            )

            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_weights)
            )

        # calculate gradients and update weights for generator
        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_weights
        )

        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_weights)
        )

        # update the augmentation probability based on the discriminator's performance
        self.augmenter.update(real_logits)
        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)
        self.real_accuracy.update_state(1.0, step(real_logits))
        self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        # track the exponential moving average of the generator's weights to decrease
        # variance in the generation quality
        for weight, ema_weight in zip(self.generator.weights, self.ema_generator.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def build_generator(self):
        def g_dense_block(layer_input, shape):
            g = Dense(shape, use_bias=False)(layer_input)
            g = BatchNormalization(scale=False)(g)
            return LeakyReLU(0.2)(g)

        def g_tconv_block(layer_input, mul):
            g = Conv2DTranspose(self.filters * mul,
                                kernel_size=4,
                                strides=2,
                                padding='same',
                                use_bias=False)(layer_input)
            g = BatchNormalization(scale=False)(g)
            return LeakyReLU(0.2)(g)

        noise = Input(shape=(self.latent_dim,))

        g = g_dense_block(noise, 8 * 8 * self.filters * 4)

        g = Reshape((8, 8, self.filters * 4))(g)

        for i in [2, 1]:
            g = g_tconv_block(g, i)

        gen_img = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh', dtype='float32')(g)

        return Model(noise, gen_img, name=f'Generator_{self.model_name}')

    def build_discriminator(self):
        def d_conv_block(layer_input, mul):
            d = Conv2D(self.filters * mul, kernel_size=4, strides=2, padding='same', use_bias=False)(layer_input)
            d = BatchNormalization(scale=False)(d)
            return LeakyReLU(0.2)(d)

        # Input img = generated image
        img = Input(shape=(self.img_size, self.img_size, 3))

        d = d_conv_block(img, 1)

        for i in [2, 4]:
            d = d_conv_block(d, i)

        d = Flatten()(d)
        d = Dropout(self.dropout_val)(d)
        validity = Dense(1, dtype='float32')(d)

        return Model(img, validity, name=f'Discriminator_{self.model_name}')


if __name__ == '__main__':

    parser = ArgumentParser(prog='Deep Convolutional GAN',
                            description='Model generates 25 human faces in test mode.')
    parser.add_argument('-t', action=BooleanOptionalAction, help='Flag for the train mode')
    parser.add_argument('-e', type=int, default=EPOCHS, help='Number of epochs to train')
    parser.add_argument('-b', type=int, default=BATCH_SIZE,
                        help='Batch size. Please mind the memory limits of your system.')
    parser.add_argument('-o', type=str, choices=['adam', 'rmsprop'], default='adam',
                        help='Optimizer. One of the following: ["adam", "rmsprop"]')
    args = parser.parse_args()

    # Creating datasets
    train_ds, test_ds = get_datasets(args.b)

    if args.o == 'adam':
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=ADAM_MOMENTUM)
    else:
        optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=5e-5)

    model = DCGAN(model_name='DC_GAN', batch_size=args.b)
    model.compile(generator_optimizer=optimizer, discriminator_optimizer=optimizer)

    # Save the best model based on the validation KID metric
    checkpoint_path = f'{MODELS_DIR}{SEP}{model.model_name}{SEP}{model.model_name}.ckpt'

    if args.t:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor="val_kid",
            mode="auto",
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )

        # Training the model
        model.fit(
            train_ds,
            epochs=args.e,
            validation_data=test_ds,
            callbacks=[
                tf.keras.callbacks.LambdaCallback(on_epoch_begin=model.update_epoch,
                                                  on_epoch_end=model.save_model_and_history,
                                                  on_batch_end=model.update_sample_images_and_history),
                checkpoint_callback,
                ],
            )

    else:
        model.load_weights(checkpoint_path).expect_partial()
        model.sample_images()
