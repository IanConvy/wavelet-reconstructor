import tensorflow as tf

# This module contains custom Keras layers used to efficiently implement 
# various wavelet operations.

# Haar basis:
haar = [2**(-0.5), 2**(-0.5)]
# Daubechies 4:
d_4 = [0.48296291314453, 0.83651630373781, 0.22414386804201, -0.12940952255126]
# Daubechies 6:
d_6 = [0.33267055295008, 0.80689150931109, 0.45987750211849, -0.13501102001025,
       -0.08544127388203, 0.03522629188571]
# Daubechies 8:
d_8 = [0.23037781330890, 0.71484657055292, 0.63088076792986, -0.02798376941686,
       -0.18703481171909, 0.03084138183556, 0.03288301166689, -0.01059740178507]

class Wavelet_Decon_Layer(tf.keras.layers.Layer):

    # This class implements a Keras layer that performs a 
    # sequence of wavelet decompositions on the input, returning
    # the scale and wavelet coefficients at each level.

    def __init__(self, h, coarse_only = False, **kwargs):
        super().__init__(**kwargs)
        self.decon_layer = Wavelet_nD_Decon(h)
        if coarse_only: # Only return the scale coefficients
            self.call_func = self.call_coarse
        else: # Return both the scale and wavelet coefficients
            self.call_func = self.call_full
        
    def build(self, input_shape):

        # The number of decompositions (depth) is given
        # by the smallest power of two that is greather
        # than or equal to the smallest dimension of the
        # input. 

        num_features = tf.cast(input_shape[1:], "float32")
        pows = tf.math.ceil(tf.math.log(num_features) / tf.math.log(2.0))
        self.depth = int(tf.reduce_min(pows))

    def call(self, inputs):
        outputs = self.call_func(inputs)
        return outputs

    def call_full(self, inputs):

        # Performs a sequence of wavelet decomposition by 
        # feeding in the scale coefficients from the previous 
        # decomposition into the decon_layer, and returns
        # the resulting scale and wavelet coefficients for
        # each decomposition.

        outputs = [inputs]
        next_input = inputs
        for _ in range(self.depth):
            decon = self.decon_layer(next_input)
            outputs.append(decon)
            decon_shape = tf.shape(decon)
            size = tf.concat([[-1], decon_shape[1:] // 2], 0)
            next_input = tf.slice(decon, tf.zeros_like(decon_shape), size)
        outputs.reverse()
        return outputs

    def call_coarse(self, inputs):

        # The same as call_full, except that only the scale
        # cofficients are returned. 

        outputs = [inputs]
        next_input = inputs
        for _ in range(self.depth):
            decon = self.decon_layer(next_input)
            decon_shape = tf.shape(decon)
            size = tf.concat([[-1], decon_shape[1:] // 2], 0)
            next_input = tf.slice(decon, tf.zeros_like(decon_shape), size)
            outputs.append(next_input)
        outputs.reverse()
        return outputs
    
class Wavelet_nD_Decon(tf.keras.layers.Layer):

    # This layer performs a single wavelet decomposition on an
    # nth-order input, which can be done by simply perfoming a 
    # wavelet decomposition along each dimension.

    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    def build(self, input_shape):
        num_dims = input_shape.rank
        self.layers = [Wavelet_Decon(self.h, i) for i in range(1, num_dims)]

    def call(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

class Wavelet_nD_Recon(tf.keras.layers.Layer):

    # This layer performs a single wavelet reconstruction on an
    # nth-order input, which can be done by simply perfoming a 
    # wavelet reconstruction along each dimension.

    def __init__(self, h, **kwargs):
        super().__init__(**kwargs)
        self.h = h

    def build(self, input_shape):
        num_dims = input_shape.rank
        self.layers = [Wavelet_Recon(self.h, i) for i in range(1, num_dims)]

    def call(self, inputs):
        outputs = inputs
        for (i, layer) in enumerate(self.layers, 1):
            decon = tf.split(outputs, 2, i)
            outputs = layer(decon)
        return outputs       
        
class Wavelet_Decon(tf.keras.layers.Layer):

    # This layer performs a wavelet decomposition along a single
    # axis, returning both the scale and wavelet coefficients.

    def __init__(self, h, axis, **kwargs):
        super().__init__(**kwargs)
        h = tf.constant(h)
        indices = tf.range(tf.size(h))
        self.h_0 = h[:, tf.newaxis, tf.newaxis]
        self.h_1 = tf.reverse(tf.dynamic_stitch([indices[::2], indices[1::2]], [-h[::2], h[1::2]]), [0])[:, tf.newaxis, tf.newaxis]
        self.basis_indices = tf.range(tf.size(h) - 1)
        self.axis = axis

    def build(self, input_shape):
        num_dims = input_shape.rank
        self.permute = tf.dynamic_stitch([tf.range(num_dims), [self.axis, num_dims-1]], [tf.range(num_dims), [num_dims-1, self.axis]])

    def call(self, inputs):
        inputs = tf.transpose(inputs, self.permute)
        cycle_indices = self.basis_indices % tf.shape(inputs)[-1]
        periodic_coeff = tf.concat([inputs, tf.gather(inputs, cycle_indices, axis = -1)], axis = -1)
        h_0_conv = tf.nn.conv1d(periodic_coeff[..., tf.newaxis], self.h_0, stride = 1, padding = "VALID")[..., 0]
        h_1_conv = tf.nn.conv1d(periodic_coeff[..., tf.newaxis], self.h_1, stride = 1, padding = "VALID")[..., 0]
        scale_coeff = tf.transpose(h_0_conv[..., ::2], self.permute)
        wavelet_coeff = tf.transpose(h_1_conv[..., ::2], self.permute)
        outputs = tf.concat([scale_coeff, wavelet_coeff], self.axis)
        return outputs

class Wavelet_Recon(tf.keras.layers.Layer):

    # This layer performs a wavelet reconstruction along a single
    # axis, returning the scale coefficients from the previous
    # decomposition.

    def __init__(self, h, axis, **kwargs):
        super().__init__(**kwargs)
        h = tf.constant(h)
        indices = tf.range(tf.size(h))
        self.h_0 = h[::-1, tf.newaxis, tf.newaxis]
        self.h_1 = tf.reverse(tf.dynamic_stitch([indices[::2], indices[1::2]], [-h[::2], h[1::2]]), [0])[::-1, tf.newaxis, tf.newaxis]
        self.basis_indices = tf.range(-(tf.size(h) - 2)//2 - 1, 0)
        self.axis = axis

    def build(self, input_shape):
        num_dims = input_shape[0].rank
        self.permute = tf.dynamic_stitch([tf.range(num_dims), [self.axis, num_dims-1]], [tf.range(num_dims), [num_dims-1, self.axis]])

    def call(self, inputs):
        inputs = tf.stack([tf.transpose(inputs[0], self.permute), tf.transpose(inputs[1], self.permute)])
        cycle_indices = self.basis_indices % tf.shape(inputs)[-1]
        new_inputs = tf.concat([tf.gather(inputs, cycle_indices, axis = -1), inputs], axis = -1)[..., 1:]
        batch_shape = tf.shape(inputs)[:-1]
        up_shape = tf.concat([batch_shape, 2*tf.shape(new_inputs)[-1:]], 0)
        coeff_up = tf.reshape(tf.stack([tf.zeros_like(new_inputs), new_inputs], -1), up_shape)
        coeff_up = tf.concat([coeff_up, tf.zeros_like(coeff_up[..., 0:1])], -1)

        scale_coeff = coeff_up[0]
        wavelet_coeff = coeff_up[1]
        h_0_conv = tf.nn.conv1d(scale_coeff[..., tf.newaxis], self.h_0, stride = 1, padding = "VALID")[..., 0]
        h_1_conv = tf.nn.conv1d(wavelet_coeff[..., tf.newaxis], self.h_1, stride = 1, padding = "VALID")[..., 0]
        outputs = tf.transpose(h_0_conv + h_1_conv, self.permute)
        return outputs

class Pad_nD(tf.keras.layers.Layer):

    # This layer applies a periodic padding transformation 
    # along each axis of the input to make it compaitible 
    # with the wavelet decomposition.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tf.cast(input_shape.as_list()[1:], "float32")
        pows = tf.math.ceil(tf.math.log(input_shape) / tf.math.log(2.0))
        pad_amounts = 2**pows - input_shape
        left_ends = tf.cast(pad_amounts // 2, "int32")
        right_starts = tf.cast(input_shape - (pad_amounts // 2 + pad_amounts % 2), "int32")
        rank = int(tf.size(left_ends)) + 1
        self.slices = []
        for i in range(1, rank):
            left_begin = [0]*rank
            right_begin = [0]*rank
            right_begin[i] = int(right_starts[i - 1])
            right_size = [-1]*rank
            left_size = [-1]*rank
            left_size[i] = int(left_ends[i - 1])
            self.slices.append((left_begin, right_begin, left_size, right_size))

    def call(self, inputs):
        outputs = inputs
        for (i, (l_begin, r_begin, l_size, r_size)) in enumerate(self.slices):
            axis = i + 1
            left_pad = tf.reverse(tf.slice(outputs, l_begin, l_size), axis = [axis])
            right_pad = tf.reverse(tf.slice(outputs, r_begin, r_size), axis = [axis])
            outputs = tf.concat([left_pad, outputs, right_pad], axis = axis)
        return outputs

class Unpad_nD(tf.keras.layers.Layer):

    # This layer applies a transformation which undoes the
    # padding applies by the Pad_nD layer.

    def __init__(self, orig_shape, **kwargs):
        super().__init__(**kwargs)
        shape_float = tf.cast(orig_shape, "float32")
        pows = tf.math.ceil(tf.math.log(shape_float) / tf.math.log(2.0))
        pad_amounts = 2**pows - shape_float
        self.starts = tf.cast(pad_amounts // 2, "int32")
        self.sizes = orig_shape
    
    def call(self, inputs):
        outputs = tf.slice(inputs, self.starts, self.sizes)
        return outputs
