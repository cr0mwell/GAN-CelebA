from base import GAN, get_datasets, step
from constants import SEP, BATCH_SIZE, EPOCHS, ADAM_MOMENTUM, MODELS_DIR

from argparse import ArgumentParser, BooleanOptionalAction

import tensorflow as tf
from tensorflow.keras.layers import (Dense, Reshape, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D,
                                     Conv2DTranspose, Conv2D, ReLU, LeakyReLU, Rescaling, Resizing, Lambda)
from tensorflow.keras import Input, Model


class WGAN(GAN):

    def __init__(self, gp_weight=10.0, **kwargs):
        super().__init__(**kwargs)

        self.gp_weight = gp_weight

    # Discriminator loss: mean(fake_logits) - mean(real_logits)
    # Gradient penalty will be added later
    @staticmethod
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Generator loss: -mean(fake_logits)
    @staticmethod
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(generator_optimizer, discriminator_optimizer, **kwargs)

        self.generator_loss_tracker = self.generator_loss
        self.discriminator_loss_tracker = self.discriminator_loss
        self.real_accuracy = tf.keras.metrics.BinaryAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [
            self.augmentation_probability_tracker,
            self.real_accuracy,
            self.kid
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0, dtype='float16')

        diff = real_images - fake_images
        interpolated = fake_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]

        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        real_images = self.augmenter(real_images, self.batch_size, training=True)

        # Training the discriminator self.d_steps steps and generator 1 step
        for i in range(self.d_steps):
            # use persistent gradient tape because gradients will be calculated twice
            with tf.GradientTape(persistent=True) as tape:
                generated_images = self.generate(self.batch_size, training=True)
                # gradient is calculated through the image augmentation
                generated_images = self.augmenter(generated_images, self.batch_size, training=True)

                # separate forward passes for the real and generated images, meaning
                # that batch normalization is applied separately
                real_logits = self.discriminator(real_images, training=True)
                generated_logits = self.discriminator(generated_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                discriminator_loss = self.discriminator_loss_tracker(real_img=real_logits, fake_img=generated_logits)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(self.batch_size, real_images, generated_images)

                # Add the gradient penalty to the original discriminator loss
                discriminator_loss = discriminator_loss + tf.cast(gp, dtype='float32') * self.gp_weight

                if i == self.d_steps - 1:
                    generator_loss = self.generator_loss_tracker(generated_logits)

            discriminator_gradients = tape.gradient(
                discriminator_loss, self.discriminator.trainable_variables
            )

            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_variables)
            )

        # calculate gradients and update weights
        generator_gradients = tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )

        # update the augmentation probability based on the discriminator's performance
        self.augmenter.update(real_logits)
        self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        self.real_accuracy.update_state(1.0, step(real_logits))
        self.augmentation_probability_tracker.update_state(self.augmenter.probability)

        # track the exponential moving average of the generator's weights to decrease
        # variance in the generation quality
        for weight, ema_weight in zip(self.generator.weights, self.ema_generator.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {"d_loss": discriminator_loss, "g_loss": generator_loss,
                self.metrics[0].name: self.metrics[0].result(),
                self.metrics[1].name: self.metrics[1].result()}

    def build_generator(self):
        def g_dense_block(layer_input, shape):
            g = Dense(shape, use_bias=False)(layer_input)
            g = BatchNormalization()(g)
            return LeakyReLU(0.2)(g)

        def g_tconv_block(input, mul, activation):
            g = Conv2DTranspose(self.filters * mul, kernel_size=4, strides=2, padding='same', use_bias=False)(input)
            g = BatchNormalization()(g)
            return activation(g)

        noise = Input(shape=(self.latent_dim,))

        g = g_dense_block(noise, 8 * 8 * self.filters * 4)
        g = Reshape((8, 8, self.filters * 4))(g)
        g = g_tconv_block(g, 2, LeakyReLU(0.2))
        g = g_tconv_block(g, 1, LeakyReLU(0.2))
        gen_img = Conv2DTranspose(3,
                                  kernel_size=4,
                                  strides=2,
                                  padding='same',
                                  activation='tanh',
                                  dtype='float32')(g)

        return Model(noise, gen_img, name=f'Generator_{self.model_name}')

    def build_discriminator(self):

        def d_conv_block(layer_input, mul):
            d = Conv2D(self.filters * mul, kernel_size=4, strides=2, padding='same')(layer_input)
            d = LeakyReLU(0.2)(d)
            return d

        # Input img = generated image
        img = Input(shape=(self.img_size, self.img_size, 3))

        d = d_conv_block(img, 1)
        d = d_conv_block(d, 2)
        d = d_conv_block(d, 4)

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

    model = WGAN(model_name='WGAN', batch_size=args.b)
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
            epochs=EPOCHS,
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
