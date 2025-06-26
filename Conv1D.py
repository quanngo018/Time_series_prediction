import tensorflow as tf

class MyConv1D(tf.keras.layers.Layer):
    def __init__(self, input_dims, filters, kernel_size, padding = 'VALID', **kwargs):
        # kwargs are passed to the parent class
        # to allow for additional configurations

        # input_dims: number of input channels aka input features
        # filters: number of filters(kernals)
        # kernel_size: size of the convolutional kernel
        # padding: 'VALID' or 'SAME'
        # 'VALID' means no padding, 'SAME' means padding to keep output size

        super(MyConv1D, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.filters = filters
        self.kernel_size = kernel_size
        self.pad = padding.upper()

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(w_init(shape=[filters, kernel_size, input_dims], dtype='float32'), trainable=True)
        self.b = tf.Variable(tf.zeros(shape=[filters], dtype='float32'), trainable=True)

        # trainable: if True, the layer's weights will be updated during training


    def call(self, inputs):
        # inputs: a 3D tensor of shape (batch_size, num_rows, input_dims)
        # where:
        # batch dimension (0) is the number of samples,
        # time dimension (1) is the number of rows (or time steps),
        # feature dimension (2) is the number of input features (channels)
        num_rows = inputs.shape[1]
        if self.pad == 'SAME':
            total_padding = self.kernel_size - 1
            num_pad_left = total_padding // 2
            num_pad_right = total_padding - num_pad_left
            # Pad the input tensor
            # no padding on the batch dimension (0),
            # padding on the time dimension (1) 
            # no padding on the feature dimension (2).
            input_padded = tf.pad(inputs, [[0, 0], [num_pad_left, num_pad_right], [0, 0]])
            num_out_rows = num_rows
        else:
            input_padded = inputs
            num_out_rows = num_rows - self.kernel_size + 1
        
        # Perform the convolution operation
        # Initialize an empty list to store the output
        # Each output will be a 2D tensor of shape (batch_size, filters)
        conv_output = []
        for i in range(num_out_rows):
            # Extract the current window of input
            input_window = input_padded[:, i:i + self.kernel_size, :]
            # Multiply the weights with the input window
            # Add new axes to both for broadcasting:
            # Resulting broadcasted shape: (batch_size, filters, kernel_size, input_dims)
            multiplied = self.w[tf.newaxis, :, :] * input_window[:, tf.newaxis, :, :]

            # Axes 2 and 3 are removed after being summed
            # This collapses the result into one scalar per filter for each batch
            # Resulting shape: (batch_size, filters)
            sum = tf.reduce_sum(multiplied, axis=[2, 3]) 

            conv_output.append(sum)

        # Stack the output list into a 3D tensor
        conv_output = tf.stack(conv_output, axis=1)
        # Add the bias term
        conv_output += self.b[tf.newaxis, tf.newaxis, :]
        # Return the output tensor
        return conv_output



        
            





            


        