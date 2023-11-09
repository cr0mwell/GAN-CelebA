import tensorflow as tf
from tensorflow.keras import mixed_precision, layers, Model, Input

import pandas as pd
import matplotlib.pyplot as plt

import cv2
import json
import os

from constants import (SAMPLES_DIR, MODELS_DIR, DATASET, SEP, DS_PATH, IMG_RES, IMG_PATH, EMA,
                       MEAN, SIGMA, LATENT_DIM, KID_IMG_RES, SAMPLE_INTERVAL, MAX_TRANSLATION, MAX_ROTATION,
                       MAX_ZOOM, BATCH_SIZE, TARGET_ACCURACY, STEPS, FILTERS, DROPOUT_VAL, DISC_STEPS)

# Setting mixed_precision global policy
mixed_precision.set_global_policy('mixed_float16')


def crop_face(import_path, export_path, target_img_size):
    """
    Resizes the images from import_path to the target_size(tuple) and saves them to the export_path.
    Filters non-fullface images.
    """

    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
                print(os.path.join(export_path, file))
                count += 1
        else:
            pass

    print(f'total: {count} / {len(files)}')


def set_configuration():
    """
    Sets Keras session configuration options
    """
    pds = tf.config.list_physical_devices('GPU')
    for gpu in pds:
        try:
            #tf.config.experimental.set_memory_growth(gpu, True)
            limit_conf = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=20000)
            tf.config.experimental.set_virtual_device_configuration(gpu, [limit_conf])
        except RuntimeError as e:
            print(e)


def add_noise(img):
    "Adds gaussian noise to the image tensor."
    # Adding noise
    return img + tf.random.normal((IMG_RES, IMG_RES, 3), mean=MEAN, stddev=SIGMA, dtype='float16')


def get_datasets(batch_size):
    "Returns two tf.data.Dataset objects (train and test datasets), split in proportion 90/10."

    # Normalizing an image to [-1, 1] range for tanh activation
    normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    # Normalizing an image to [-1, 1] range for sigmoid activation
    #normalization_layer = tf.keras.layers.Rescaling(1./255)

    # Resizing dataset source images to the target IMG_RES size if necessary
    if not os.path.exists(IMG_PATH):
        if os.path.exists(src_path := f'{DS_PATH}{SEP}original'):
            os.makedirs(f'{DS_PATH}{SEP}{IMG_RES}x{IMG_RES}', exist_ok=True)
            crop_face(src_path, IMG_PATH, (IMG_RES, IMG_RES))
        else:
            raise FileNotFoundError(f'Dataset source directory doesn\'t exist: {src_path}')

    train_ds = tf.keras.utils.image_dataset_from_directory(IMG_PATH,
                                                           labels=None,
                                                           validation_split=0.15,
                                                           subset="training",
                                                           seed=123,
                                                           image_size=(IMG_RES, IMG_RES),
                                                           batch_size=None) \
        .batch(batch_size, drop_remainder=True) \
        .map(lambda x: normalization_layer(x), num_parallel_calls=tf.data.AUTOTUNE) \
        .map(add_noise, num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(IMG_PATH,
                                                          labels=None,
                                                          validation_split=0.15,
                                                          subset="validation",
                                                          seed=123,
                                                          image_size=(IMG_RES, IMG_RES),
                                                          batch_size=None) \
        .batch(batch_size, drop_remainder=True) \
        .map(lambda x: normalization_layer(x), num_parallel_calls=tf.data.AUTOTUNE) \
        .cache() \
        .prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, test_ds


def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + tf.sign(values))


class AdaptiveAugmenter(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # stores the current probability of an image being augmented
        self.probability = tf.Variable(0.0, dtype='float16')

        # blitting and geometric augmentations are the most helpful in the low-data regime
        self.augmenter = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(IMG_RES, IMG_RES, 3)),
                # blitting/x-flip:
                layers.RandomFlip("horizontal"),
                # blitting/integer translation:
                layers.RandomTranslation(
                    height_factor=MAX_TRANSLATION,
                    width_factor=MAX_TRANSLATION,
                    interpolation="nearest",
                ),
                # geometric/rotation:
                layers.RandomRotation(factor=MAX_ROTATION),
                # geometric/isotropic and anisotropic scaling:
                layers.RandomZoom(
                    height_factor=(-MAX_ZOOM, 0.0), width_factor=(-MAX_ZOOM, 0.0)
                ),
            ],
            name="adaptive_augmenter",
        )

    def call(self, images, size, training):
        if training:
            augmented_images = self.augmenter(images, training)

            # during training either the original or the augmented images are selected
            # based on self.probability
            augmentation_values = tf.random.uniform(
                shape=(size, 1, 1, 1), minval=0.0, maxval=1.0, dtype='float16'
            )
            augmentation_bools = tf.math.less(augmentation_values, self.probability)

            images = tf.where(augmentation_bools, augmented_images, images)
        return images

    def update(self, real_logits):
        current_accuracy = tf.reduce_mean(step(real_logits))

        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - TARGET_ACCURACY
        self.probability.assign(
            tf.clip_by_value(
                self.probability + tf.cast(accuracy_error, dtype='float16') / STEPS, 0.0, 1.0
            )
        )


class KID(tf.keras.metrics.Metric):
    def __init__(self, name="kid", **kwargs):
        super().__init__(name=name, **kwargs)

        # KID is estimated per batch and is averaged across batches
        self.kid_tracker = tf.keras.metrics.Mean()

        # a pretrained InceptionV3 is used without its classification layer
        # transform the pixel values to the 0-255 range, then use the same
        # preprocessing as during pretraining
        self.encoder = tf.keras.Sequential(
            [
                Input(shape=(IMG_RES, IMG_RES, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=KID_IMG_RES, width=KID_IMG_RES),
                layers.Lambda(tf.keras.applications.inception_v3.preprocess_input),
                tf.keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(KID_IMG_RES, KID_IMG_RES, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype='float16')
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype='float16')
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size, dtype='float16'))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size, dtype='float16'))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


class GAN(tf.keras.Model):
    """
    Base class for GAN models.

    :param dataset: String. Name of the dataset that will be used to train the models.
    :param model_name: String. Name of the model
    :param img_size: Int. size of the images from the train dataset
    :param latent_dim: Int. Latent dimension size
    :param filters: Int. Filter size
    :param dropout_val: Float. Dropout probability
    :param sample_interval: Int. Generator image sampling interval in training iterations
    :param batch_size: Int. Batch size for the training iteration
    :param d_steps: Int. Discriminator training iterations per batch
    :param ema: Float. Ema parameter for the exponential smoothing.
    """
    def __init__(self, img_size=IMG_RES, latent_dim=LATENT_DIM, dataset=DATASET, batch_size=BATCH_SIZE,
                 ema=EMA, filters=FILTERS, dropout_val=DROPOUT_VAL, sample_interval=SAMPLE_INTERVAL,
                 d_steps=DISC_STEPS, model_name='base_GAN'):

        super().__init__()
        self.loss_history = []

        # Model parameters
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.dataset_name = dataset
        self.filters = filters
        self.dropout_val = dropout_val
        self.ema = ema
        self.batch_size = batch_size
        self.sample_interval = sample_interval
        self.model_name = model_name
        self.epoch = 0  # for epochs tracking
        self.d_steps = d_steps

        # Creating models
        self.augmenter = AdaptiveAugmenter()
        self.generator = self.build_generator()
        self.ema_generator = tf.keras.models.clone_model(self.generator)
        self.discriminator = self.build_discriminator()

        # Creating model folders
        os.makedirs(f'{MODELS_DIR}{SEP}{self.model_name}', exist_ok=True)
        os.makedirs(f'{SAMPLES_DIR}{SEP}{self.model_name}', exist_ok=True)

        # Loading loss history
        history_file = f'{MODELS_DIR}{SEP}{self.model_name}{SEP}history.json'
        if os.path.exists(history_file):
            with open(history_file) as f:
                self.loss_history = json.load(f)
                print("Successfully loaded history")
        else:
            print(f'Failed to load loss history: no such file "{history_file}"')

        print('---------------------generator summary----------------------------')
        self.generator.summary()

        print('---------------------discriminator summary----------------------------')
        self.discriminator.summary()

    def build_generator(self):
        raise NotImplementedError('This method should be implemented in the subclass')

    def build_discriminator(self):
        raise NotImplementedError('This method should be implemented in the subclass')

    def compile(self, generator_optimizer, discriminator_optimizer, **kwargs):
        super().compile(**kwargs)

        # separate optimizers for the two networks
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.augmentation_probability_tracker = tf.keras.metrics.Mean(name="aug_p")
        self.kid = KID()

    def generate(self, batch_size, training):
        latent_samples = tf.random.normal(shape=(batch_size, LATENT_DIM))
        # use ema_generator during inference
        if training:
            generated_images = self.generator(latent_samples, training)
        else:
            generated_images = self.ema_generator(latent_samples, training)
        return generated_images

    def train_step(self, real_images):
        raise NotImplementedError('This method should be implemented in the subclass')

    def test_step(self, real_images):
        generated_images = self.generate(self.batch_size, training=False)
        self.kid.update_state(tf.cast(real_images, dtype='float32'), generated_images)

        # only KID is measured during the evaluation phase for computational efficiency
        return {self.kid.name: self.kid.result()}

    def update_epoch(self, epoch, logs):
        self.epoch = epoch

    def save_model_and_history(self, epoch, logs):
        """
        Plots and saves the loss history graph at the epoch's end.
        """
        hist = pd.DataFrame(self.loss_history)
        plt.figure(figsize=(20, 5))
        for column in ['D', 'G']:
            plt.plot(hist[column], label=column)
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.savefig(f'{SAMPLES_DIR}{SEP}{self.model_name}{SEP}Loss history.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Saving history state
        with open(f'{MODELS_DIR}{SEP}{self.model_name}{SEP}history.json', 'w') as f:
            json.dump(self.loss_history, f)

        # Saving the model
        if epoch and epoch % 5 == 0:
            checkpoint_path = f'{MODELS_DIR}{SEP}{self.model_name}'
            self.generator.save_weights(f'{checkpoint_path}{SEP}{self.model_name}_generator.tf',
                                        overwrite=True, save_format='tf')
            self.discriminator.save_weights(f'{checkpoint_path}{SEP}{self.model_name}_discriminator.tf',
                                            overwrite=True, save_format='tf')

    def sample_images(self, iteration=0, training=False):
        # Sampling generator images
        r, c = 5, 5
        gen_imgs = self.generate(25, training=False).numpy()

        # Rescale images 0 - 1
        gen_imgs = gen_imgs/2. + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col in range(c):
                axs[row, col].imshow(gen_imgs[cnt])
                axs[row, col].axis('off')
                cnt += 1

        if training:
            # Save generated images and the high resolution originals
            fig.savefig(f'{SAMPLES_DIR}/{self.model_name}/e{self.epoch}-i{iteration}.png',
                        bbox_inches='tight', pad_inches=0, dpi=100)
        else:
            plt.show()
        plt.close()

    def update_sample_images_and_history(self, iteration, logs):
        """
        Saves sampled images generated by generator every SAMPLE_INTERVAL iterations.
        Every iteration adds loss history.
        """
        # Updating history
        self.loss_history.append({'D': logs['d_loss'], 'G': logs['g_loss']})

        if iteration % self.sample_interval == 0:
            self.sample_images(iteration, training=True)


if __name__ == '__main__':
    # Resizing the dataset images
    os.makedirs(f'{DS_PATH}{SEP}{IMG_RES}x{IMG_RES}', exist_ok=True)
    crop_face(DS_PATH, f'{DS_PATH}{SEP}{IMG_RES}x{IMG_RES}', (IMG_RES, IMG_RES))
