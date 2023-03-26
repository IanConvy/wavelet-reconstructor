import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

import wavelet

# This module defines the wavelet decomposition models and
# runs numerical test.

class Wavelet():

    # This model is trained to generate the wavelet coefficients
    # needed to reconstruct an input given its scale coefficients.
    # This allows the model to perform reconstruction on coarse-grained
    # data even if the corresponding wavelet coefficients are discarded. 

    def __init__(self, h):
        self.pad_layer = wavelet.Pad_nD()
        self.decon_layer = wavelet.Wavelet_Decon_Layer(h, coarse_only = True)
        self.recon_layer = wavelet.Wavelet_nD_Recon(h)

    def load(self):
        self.cores = [tf.keras.models.load_model(f"mnist_models/wavelet/core_{2**i}") for i in range(5)]
        self.models = [self.assemble_model(core, 2**i) for (i, core) in enumerate(self.cores)]
        self.cleaner = tf.keras.models.load_model("mnist_models/wavelet/cleaner_4")

    def build(self):

        # Each of the model cores correspond to a different
        # coarse-grained input size. The 1x1 and 2x2 models
        # are dense networks, while the remaining models
        # are convolutional. 

        self.models = []
        self.cores = []

        model_1_core = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(3),
            tf.keras.layers.Reshape([1, 1, 3])
        ])
        model_1 = self.assemble_model(model_1_core, 1)
        self.cores.append(model_1_core)
        self.models.append(model_1)

        model_2_core = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dense(32, "relu"),
            tf.keras.layers.Dense(12),
            tf.keras.layers.Reshape([2, 2, 3])
        ])
        model_2 = self.assemble_model(model_2_core, 2)
        self.cores.append(model_2_core)
        self.models.append(model_2)

        model_4_core = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(3, 3, padding = "same")
        ])
        model_4 = self.assemble_model(model_4_core, 4)
        self.cores.append(model_4_core)
        self.models.append(model_4)
        
        model_8_core = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(3, 3, padding = "same")
        ])
        model_8 = self.assemble_model(model_8_core, 8)
        self.cores.append(model_8_core)
        self.models.append(model_8)

        model_16_core = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(3, 3, padding = "same")
        ])
        model_16 = self.assemble_model(model_16_core, 16)
        self.cores.append(model_16_core)
        self.models.append(model_16)

    def train_all(self, train_images, test_images, epochs, batch_size, save = False):
        for (i, model) in enumerate(self.models):
            length = 2**i
            print(f"Training model_{length}")
            self.train(length, train_images, test_images, epochs, batch_size, save)

    def train(self, length, train_images, test_images, epochs, batch_size, save = False):

        # Each model is trained using the exact decomposition coefficients, rather than the
        # reconstructed output from the previous layers.

        train_data = self.wavelet_transform(train_images.shuffle(60000).batch(batch_size), length)
        test_data = self.wavelet_transform(test_images.batch(batch_size), length)
        index = int(np.math.log2(length))
        model = self.models[index]

        model.compile(
            loss = "mae",
            optimizer = tf.keras.optimizers.RMSprop(0.0005))
        model.fit(
            train_data,
            epochs = epochs, 
            validation_data = test_data,
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights = True)])
        if save:
            core = self.cores[index]
            core.save(f"mnist_models/wavelet/core_{length}")

    def train_cleaner(self, length, train_images, test_images, epochs, batch_size, save = False):

        # The cleaner is a part of the model that finesses the final
        # reconstruction so that it more closely matches the expected
        # profile of the dataset. It is optimized using the output from
        # the previously trained layers.

        self.cleaner = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, -1)),
            tf.keras.layers.Conv2D(128, 3, padding = "same", activation = "relu"),
            tf.keras.layers.Conv2D(1, 3, activation = "sigmoid", padding = "same"),
            tf.keras.layers.Reshape([32, 32])
        ])
        first = tf.keras.layers.Input([length, length])
        recon = self.reconstruct(first, length)
        cleaned = self.cleaner(recon)

        model = tf.keras.Model(inputs = [first], outputs = [cleaned])
        for core in self.cores:
            core.trainable = False
        model.compile(
            loss = "mae",
            optimizer = tf.keras.optimizers.RMSprop(0.0005))
        
        train_data = self.wavelet_transform(train_images.shuffle(60000).batch(batch_size), length, True)
        test_data = self.wavelet_transform(test_images.batch(batch_size), length, True)
        model.fit(
            train_data,
            epochs = epochs, 
            validation_data = test_data,
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10, restore_best_weights = True)])
        if save:
            self.cleaner.save(f"mnist_models/wavelet/cleaner_{length}")

    def assemble_model(self, core, length):

        # This function adds the wavelet reconstruction layer
        # to the model core.

        first = tf.keras.layers.Input([length, length])
        fine = core(first)
        combined = tf.keras.layers.Lambda(
            lambda x: tf.concat([
                tf.concat([x[0], x[1][..., 0]], axis = 2),
                tf.concat([x[1][..., 1], x[1][..., 2]], axis = 2)
            ], axis = 1)
        )([first, fine])
        recon = self.recon_layer(combined)
        model = tf.keras.Model(inputs = [first], outputs = [recon])
        return model

    def wavelet_transform(self, dataset, length, keep_orig = False):

        # This function carries out a wavelet decomposition on the
        # inputs using the decon_layer.

        index = int(np.math.log2(length))
        padded = dataset.map(lambda x: self.pad_layer(x))
        if keep_orig:
            coeff = padded.map(lambda y: (self.decon_layer(y)[index], y))
        else:
            coeff = padded.map(lambda y: (self.decon_layer(y)[index], self.decon_layer(y)[index + 1]))
        return coeff

    def reconstruct(self, inputs, length = None):

        # This function uses the model to appoximately reconstructs 
        # the original input based on its coarse-grained scale 
        # coefficients.

        if length is None:
            length = tf.shape(inputs)[1]
        start_index = int(np.math.log2(length))
        next_input = inputs
        for i in range(start_index, 5):
            model = self.models[i]
            next_input = model(next_input)
        return next_input

    def clean(self, inputs):
        cleaned = self.cleaner(inputs)
        return cleaned

    def sample(self, length, num_samples):

        # This function samples scale cofficients from a 
        # uniform distribution and then generates an "original" 
        # input using the model.

        uniform = tf.random.uniform([num_samples, length, length])
        reconstruct = self.reconstruct(uniform)
        return reconstruct

    def sample_gaussian(self, dataset, length, num_samples):

        # This function uses the network as a generative model
        # by fitting a Gaussian distribution to the scale coefficients
        # of the decomposed dataset and then generating new images
        # from it.

        decon = self.wavelet_transform(dataset.batch(60000), length)
        data = tf.reshape(next(iter(decon))[0], [60000, -1])
        cov = tfp.stats.covariance(data)
        mean = tf.reduce_mean(data, 0)
        samples = tfp.distributions.MultivariateNormalFullCovariance(mean, cov).sample([num_samples])
        recon = self.reconstruct(tf.reshape(samples, [num_samples, length, length]))
        cleaned = self.cleaner(recon)
        return cleaned

def get_images():

    # This function retrieves the MNIST dataset.

    ((train_data), train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = tf.data.Dataset.from_tensor_slices(train_data).map(lambda x: tf.cast(x, "float32") / 255)
    test_images = tf.data.Dataset.from_tensor_slices(test_data).map(lambda x: tf.cast(x, "float32") / 255)
    return (train_images, test_images)

def run_similarity_trial(model, train_images, num_samples = 100):

    # This function generates samples from a generative model and
    # then matches it with the closest example from the training set.

    orig_images = next(iter(train_images.batch(60000))).numpy()
    sampled_images = model.sample(num_samples).numpy()
    sqr_diff = (orig_images[None] - sampled_images[:, None])**2
    closest_index = np.argmin(sqr_diff.sum(axis = (-1, -2)), 1)
    closest_images = orig_images[closest_index]
    for (sampled_image, closest_image) in zip(sampled_images, closest_images):
        (fig, (ax_1, ax_2)) = plt.subplots(1, 2)
        ax_1.imshow(sampled_image, cmap = "gray")
        ax_2.imshow(closest_image, cmap = "gray")
        plt.show()

def clip_pixels(inputs):

    # This function clips pixels to be in the range [0, 1].

    lower_bound = tf.where(inputs > 0, inputs, tf.zeros_like(inputs))
    clipped = tf.where(lower_bound < 1, lower_bound, tf.ones_like(inputs))
    return clipped

# The following code trains (or loads) a reconstruction
# model and then evaluates its performance on the test set.

(train_images, test_images) = get_images()

batch_size = 32
epochs = 10

model = Wavelet(wavelet.d_4)
model.build()
model.train_all(train_images, test_images, epochs, batch_size, True)
# model.load()
model.train_cleaner(2, train_images, test_images, epochs, batch_size, True)

for ((decon, _), orig) in tf.data.Dataset.zip((model.wavelet_transform(test_images.batch(1), 4), test_images)):
    recon = model.reconstruct(decon)[0]
    clipped = clip_pixels(recon)[2:30, 2:30]
    (fig, (ax_1, ax_2, ax_3)) = plt.subplots(1, 3)
    ax_1.imshow(clipped, cmap = "gray")
    ax_2.imshow(model.clean(recon[None])[0, 2:30, 2:30], cmap = "gray")
    ax_3.imshow(orig, cmap = "gray")
    plt.show()
