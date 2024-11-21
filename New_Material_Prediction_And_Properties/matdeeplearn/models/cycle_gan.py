import tensorflow as tf
from typing import List, Tuple, Optional


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int):
        super().__init__()
        self.conv_block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.LayerNormalization()
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.conv_block(x)


class MaterialGenerator(tf.keras.Model):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            filters: int = 64,
            n_blocks: int = 9
    ):
        super().__init__()

        # Initial convolution
        self.initial = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 7, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation('relu')
        ])

        # Downsampling
        self.downsample = []
        in_filters = filters
        for i in range(2):
            out_filters = in_filters * 2
            self.downsample.append(tf.keras.Sequential([
                tf.keras.layers.Conv2D(
                    out_filters, 3, strides=2, padding='same'
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation('relu')
            ]))
            in_filters = out_filters

        # Residual blocks
        self.res_blocks = [
            ResidualBlock(in_filters) for _ in range(n_blocks)
        ]

        # Upsampling
        self.upsample = []
        for i in range(2):
            out_filters = in_filters // 2
            self.upsample.append(tf.keras.Sequential([
                tf.keras.layers.Conv2DTranspose(
                    out_filters, 3, strides=2, padding='same'
                ),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Activation('relu')
            ]))
            in_filters = out_filters

        # Output layer
        self.output_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(output_channels, 7, padding='same'),
            tf.keras.layers.Activation('tanh')
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.initial(x)

        # Downsample
        for layer in self.downsample:
            x = layer(x)

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Upsample
        for layer in self.upsample:
            x = layer(x)

        return self.output_layer(x)


class MaterialDiscriminator(tf.keras.Model):
    def __init__(self, input_channels: int):
        super().__init__()

        self.model = tf.keras.Sequential([
            # Initial conv
            tf.keras.layers.Conv2D(64, 4, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2),

            # Intermediate layers
            tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv2D(256, 4, strides=2, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            tf.keras.layers.Conv2D(512, 4, padding='same'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LeakyReLU(0.2),

            # Output layer
            tf.keras.layers.Conv2D(1, 4, padding='same')
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.model(x)
        return tf.keras.layers.GlobalAveragePooling2D()(x)


class MaterialCycleGAN(tf.keras.Model):
    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            lambda_cycle: float = 10.0,
            lambda_identity: float = 0.5
    ):
        super().__init__()

        self.gen_A = MaterialGenerator(input_shape[-1], input_shape[-1])
        self.gen_B = MaterialGenerator(input_shape[-1], input_shape[-1])
        self.disc_A = MaterialDiscriminator(input_shape[-1])
        self.disc_B = MaterialDiscriminator(input_shape[-1])

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    @tf.function
    def train_step(
            self,
            real_A: tf.Tensor,
            real_B: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake samples
            fake_B = self.gen_A(real_A, training=True)
            fake_A = self.gen_B(real_B, training=True)

            # Cycle consistency
            cycled_A = self.gen_B(fake_B, training=True)
            cycled_B = self.gen_A(fake_A, training=True)

            # Identity mapping
            same_A = self.gen_B(real_A, training=True)
            same_B = self.gen_A(real_B, training=True)

            # Discriminator outputs
            disc_real_A = self.disc_A(real_A, training=True)
            disc_fake_A = self.disc_A(fake_A, training=True)
            disc_real_B = self.disc_B(real_B, training=True)
            disc_fake_B = self.disc_B(fake_B, training=True)

            # Generator losses
            gen_A_loss = self.generator_loss(disc_fake_B)
            gen_B_loss = self.generator_loss(disc_fake_A)

            # Cycle consistency losses
            cycle_A_loss = self.cycle_loss(real_A, cycled_A)
            cycle_B_loss = self.cycle_loss(real_B, cycled_B)
            total_cycle_loss = cycle_A_loss + cycle_B_loss

            # Identity losses
            identity_A_loss = self.identity_loss(real_A, same_A)
            identity_B_loss = self.identity_loss(real_B, same_B)

            # Total generator losses
            total_gen_A_loss = gen_A_loss + \
                               self.lambda_cycle * total_cycle_loss + \
                               self.lambda_identity * identity_A_loss

            total_gen_B_loss = gen_B_loss + \
                               self.lambda_cycle * total_cycle_loss + \
                               self.lambda_identity * identity_B_loss

            # Discriminator losses
            disc_A_loss = self.discriminator_loss(disc_real_A, disc_fake_A)
            disc_B_loss = self.discriminator_loss(disc_real_B, disc_fake_B)

        # Calculate gradients and apply updates
        gen_A_gradients = tape.gradient(
            total_gen_A_loss, self.gen_A.trainable_variables
        )
        gen_B_gradients = tape.gradient(
            total_gen_B_loss, self.gen_B.trainable_variables
        )
        disc_A_gradients = tape.gradient(
            disc_A_loss, self.disc_A.trainable_variables
        )
        disc_B_gradients = tape.gradient(
            disc_B_loss, self.disc_B.trainable_variables
        )

        self.gen_opt.apply_gradients(
            zip(gen_A_gradients, self.gen_A.trainable_variables)
        )
        self.gen_opt.apply_gradients(
            zip(gen_B_gradients, self.gen_B.trainable_variables)
        )
        self.disc_opt.apply_gradients(
            zip(disc_A_gradients, self.disc_A.trainable_variables)
        )
        self.disc_opt.apply_gradients(
            zip(disc_B_gradients, self.disc_B.trainable_variables)
        )

        return {
            'gen_A_loss': total_gen_A_loss,
            'gen_B_loss': total_gen_B_loss,
            'disc_A_loss': disc_A_loss,
            'disc_B_loss': disc_B_loss
        }

    def generator_loss(self, disc_fake: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.math.squared_difference(disc_fake, 1))

    def discriminator_loss(
            self,
            disc_real: tf.Tensor,
            disc_fake: tf.Tensor
    ) -> tf.Tensor:
        real_loss = tf.reduce_mean(tf.math.squared_difference(disc_real, 1))
        fake_loss = tf.reduce_mean(tf.math.squared_difference(disc_fake, 0))
        return (real_loss + fake_loss) * 0.5

    def cycle_loss(
            self,
            real: tf.Tensor,
            cycled: tf.Tensor
    ) -> tf.Tensor:
        return tf.reduce_mean(tf.abs(real - cycled))

    def identity_loss(
            self,
            real: tf.Tensor,
            same: tf.Tensor
    ) -> tf.Tensor:
        return tf.reduce_mean(tf.abs(real - same))