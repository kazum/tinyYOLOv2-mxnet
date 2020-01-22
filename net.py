import mxnet as mx
import numpy as np

bn_epsilon = 1e-3
relu_alpha = 0.1
dtype = 'float32'


def create_network():
    x = mx.sym.Variable('data', dtype=dtype)

    #1 conv1     16  3 x 3 / 1   416 x 416 x   3   ->   416 x 416 x  16
    h1 = mx.sym.Convolution(x, kernel=(3,3), pad=(1,1), num_filter=16, name="conv1", no_bias=False)
    o1 = mx.sym.LeakyReLU(h1, slope=relu_alpha, act_type="leaky")
    
    # IMPORTANT: YOLOv2 sets the biases in every convolution = 0 and keeps only the betas (offsets) of the Batch Normalization!
    # So in the end there will be only mean,var,beta(offset),gamma(scale) for every single output channel!
    
    #2 max1          2 x 2 / 2   416 x 416 x  16   ->   208 x 208 x  16
    max1 = mx.sym.Pooling(o1, kernel=(2,2), stride=(2,2), pool_type='max')
    
    #3 conv2     32  3 x 3 / 1   208 x 208 x  16   ->   208 x 208 x  32
    h2 = mx.sym.Convolution(max1, kernel=(3,3), pad=(1,1), num_filter=32, name='conv2', no_bias=False)
    o2 = mx.sym.LeakyReLU(h2, slope=relu_alpha, act_type="leaky")

    #4 max2          2 x 2 / 2   208 x 208 x  32   ->   104 x 104 x  32
    max2 = mx.sym.Pooling(o2, kernel=(2,2), stride=(2,2), pool_type='max')

    #5 conv3     64  3 x 3 / 1   104 x 104 x  32   ->   104 x 104 x  64
    h3 = mx.sym.Convolution(max2, kernel=(3,3), pad=(1,1), num_filter=64, name='conv3', no_bias=False)
    o3 = mx.sym.LeakyReLU(h3, slope=relu_alpha, act_type="leaky")
    
    #6 max3          2 x 2 / 2   104 x 104 x  64   ->    52 x  52 x  64
    max3 = mx.sym.Pooling(o3, kernel=(2,2), stride=(2,2), pool_type='max')
    
    #7 conv4    128  3 x 3 / 1    52 x  52 x  64   ->    52 x  52 x 128
    h4 = mx.sym.Convolution(max3, kernel=(3,3), pad=(1,1), num_filter=128, name='conv4', no_bias=False)
    o4 = mx.sym.LeakyReLU(h4, slope=relu_alpha, act_type="leaky")
    
    #8 max4          2 x 2 / 2    52 x  52 x 128   ->    26 x  26 x 128
    max4 = mx.sym.Pooling(o4, kernel=(2,2), stride=(2,2), pool_type='max')
    
    #9 conv5    256  3 x 3 / 1    26 x  26 x 128   ->    26 x  26 x 256
    h5 = mx.sym.Convolution(max4, kernel=(3,3), pad=(1,1), num_filter=256, name='conv5', no_bias=False)
    o5 = mx.sym.LeakyReLU(h5, slope=relu_alpha, act_type="leaky")
    
    #10 max5          2 x 2 / 2    26 x  26 x 256   ->    13 x  13 x 256
    max5 = mx.sym.Pooling(o5, kernel=(2,2), stride=(2,2), pool_type='max')
    
    #11 conv6   512  3 x 3 / 1    13 x  13 x 256   ->    13 x  13 512
    h6 = mx.sym.Convolution(max5, kernel=(3,3), pad=(1,1), num_filter=512, name='conv6', no_bias=False)
    o6 = mx.sym.LeakyReLU(h6, slope=relu_alpha, act_type="leaky")
    
    #12 max6          2 x 2 / 1    13 x  13 x 512   ->    13 x  13 x 512
    max6 = mx.sym.Pooling(o6, kernel=(2,2), stride=(1,1), pad=(1,1), pool_type='max')
    max6 = mx.sym.slice(max6, begin=(0,0,1,1), end=(None,None,None,None))
    
    #13 conv7    1024  1 x 1 / 1    13 x  13 x512   ->    13 x  13 x 1024
    h7 = mx.sym.Convolution(max6, kernel=(3,3), pad=(1,1), num_filter=1024, name='conv7', no_bias=False)
    o7 = mx.sym.LeakyReLU(h7, slope=relu_alpha, act_type="leaky")
    
    #14 conv8   1024  3 x 3 / 1    13 x 13 x 1024   ->    13 x  13 x1024
    h8 = mx.sym.Convolution(o7, kernel=(3,3), pad=(1,1), num_filter=1024, name='conv8', no_bias=False)
    o8 = mx.sym.LeakyReLU(h8, slope=relu_alpha, act_type="leaky")
    
    #15 conv9   125  1 x 1 / 1    13 x  13 x 1024   ->    13 x  13 x125
    h9 = mx.sym.Convolution(o8, kernel=(1,1), num_filter=125, name='conv9', no_bias=False)
    # Linear output!
    o9 = h9

    return o9


# IMPORTANT: Weights order in the binary file is [ 'biases','gamma','moving_mean','moving_variance','kernel']
# IMPORTANT: biases ARE NOT the usual biases to add after the conv2d! They refer to the betas (offsets) in the Batch Normalization!
# IMPORTANT: the biases added after the conv2d are set to zero! 
# IMPORTANT: to use the weights they actually need to be de-normalized because of the Batch Normalization! ( see later )

def load_conv_layer_bn(name, loaded_weights, shape, offset):
    # Conv layer with Batch norm

    n_kernel_weights = shape[0] * shape[1] * shape[2] * shape[3]
    n_output_channels = shape[-1]
    n_bn_mean = n_output_channels
    n_bn_var = n_output_channels
    n_biases = n_output_channels
    n_bn_gamma = n_output_channels

    n_weights_conv_bn = (n_kernel_weights + n_output_channels * 4)

    # IMPORTANT: This is where (n_kernel_weights + n_output_channels * 4) comes from: 
    # n_params = kernel_shape + n_biases + n_bn_means + n_bn_var + n_bn_gammas
    # n_params = kernel_shape + n_biases + n_output_channels + n_output_channels + n_output_channels
    # n_params = kernel_shape + n_output_channels + n_output_channels + n_output_channels + n_output_channels
    # n_params = kernel_shape + n_output_channels*4
    # IMPORTANT: YOLOv2 sets the biases in every convolution = 0 and keeps only the betas (offsets) of the Batch Normalization!
    # So in the end there will be only mean,var,beta(offset),gamma(scale) for every single output channel!

    print('Loading '+str(n_weights_conv_bn)+' weights of '+name+' ...')

    biases = loaded_weights[offset:offset+n_biases]
    offset = offset + n_biases
    gammas = loaded_weights[offset:offset+n_bn_gamma]
    offset = offset + n_bn_gamma
    means = loaded_weights[offset:offset+n_bn_mean]
    offset = offset + n_bn_mean
    var = loaded_weights[offset:offset+n_bn_var]
    offset = offset + n_bn_var
    kernel_weights = loaded_weights[offset:offset+n_kernel_weights]
    offset = offset + n_kernel_weights

    # IMPORTANT: DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
    kernel_weights = np.reshape(kernel_weights,(shape[3],shape[2],shape[0],shape[1]),order='C')

    # IMPORTANT: Denormalize the weights with the Batch Normalization parameters
    for i in range(n_output_channels):

        scale = gammas[i] / np.sqrt(var[i] + bn_epsilon)
        kernel_weights[i,:,:,:] = kernel_weights[i,:,:,:] * scale
        biases[i] = biases[i] - means[i] * scale

    return biases, kernel_weights, offset


def load_conv_layer(name, loaded_weights, shape, offset):
    # Conv layer without Batch norm

    n_kernel_weights = shape[0]*shape[1]*shape[2]*shape[3]
    n_output_channels = shape[-1]
    n_biases = n_output_channels

    n_weights_conv = (n_kernel_weights + n_output_channels)
    # The number of weights is a conv layer without batchnorm is: (kernel_height*kernel_width + n_biases)
    print('Loading '+str(n_weights_conv)+' weights of '+name+' ...')

    biases = loaded_weights[offset:offset+n_biases]
    offset = offset + n_biases
    kernel_weights = loaded_weights[offset:offset+n_kernel_weights]
    offset = offset + n_kernel_weights

    # IMPORTANT: DarkNet conv_weights are serialized Caffe-style: (out_dim, in_dim, height, width)
    kernel_weights = np.reshape(kernel_weights,(shape[3],shape[2],shape[0],shape[1]),order='C')

    return biases,kernel_weights,offset


def load_weight(weights_path):
    args = {}

    # Load the binary to an array of float32
    loaded_weights = []
    loaded_weights = np.fromfile(weights_path, dtype='f')

    # Delete the first 4 that are not real params...
    loaded_weights = loaded_weights[4:]

    print('Total number of params to load = {}'.format(len(loaded_weights)))

    # IMPORTANT: starting from offset=0, layer by layer, we will get the exact number of parameters required and assign them!

    # Conv1 , 3x3, 3->16
    offset = 0
    biases, kernel_weights, offset = load_conv_layer_bn('conv1', loaded_weights, [3,3,3,16], offset)
    args['conv1_bias'] = mx.nd.array(biases)
    args['conv1_weight'] = mx.nd.array(kernel_weights)

    # Conv2 , 3x3, 16->32
    biases, kernel_weights, offset = load_conv_layer_bn('conv2', loaded_weights, [3,3,16,32], offset)
    args['conv2_bias'] = mx.nd.array(biases)
    args['conv2_weight'] = mx.nd.array(kernel_weights)
    
    # Conv3 , 3x3, 32->64
    biases, kernel_weights, offset = load_conv_layer_bn('conv3', loaded_weights, [3,3,32,64], offset)
    args['conv3_bias'] = mx.nd.array(biases)
    args['conv3_weight'] = mx.nd.array(kernel_weights)

    # Conv4 , 3x3, 64->128
    biases, kernel_weights, offset = load_conv_layer_bn('conv4', loaded_weights,[3,3,64,128], offset)
    args['conv4_bias'] = mx.nd.array(biases)
    args['conv4_weight'] = mx.nd.array(kernel_weights)

    # Conv5 , 3x3, 128->256
    biases, kernel_weights, offset = load_conv_layer_bn('conv5', loaded_weights,[3,3,128,256], offset)
    args['conv5_bias'] = mx.nd.array(biases)
    args['conv5_weight'] = mx.nd.array(kernel_weights)

    # Conv6 , 3x3, 256->512
    biases, kernel_weights, offset = load_conv_layer_bn('conv6', loaded_weights,[3,3,256,512], offset)
    args['conv6_bias'] = mx.nd.array(biases)
    args['conv6_weight'] = mx.nd.array(kernel_weights)

    # Conv7 , 3x3, 512->1024
    biases, kernel_weights, offset = load_conv_layer_bn('conv7', loaded_weights,[3,3,512,1024], offset)
    args['conv7_bias'] = mx.nd.array(biases)
    args['conv7_weight'] = mx.nd.array(kernel_weights)

    # Conv8 , 3x3, 1024->1024
    biases, kernel_weights, offset = load_conv_layer_bn('conv8', loaded_weights,[3,3,1024,1024], offset)
    args['conv8_bias'] = mx.nd.array(biases)
    args['conv8_weight'] = mx.nd.array(kernel_weights)

    # Conv9 , 1x1, 1024->125
    biases, kernel_weights, offset = load_conv_layer('conv9', loaded_weights,[1,1,1024,125], offset)
    args['conv9_bias'] = mx.nd.array(biases)
    args['conv9_weight'] = mx.nd.array(kernel_weights)

    # These two numbers MUST be equal! 
    print('Final offset = {}'.format(offset))
    print('Total number of params in the weight file = {}'.format(len(loaded_weights)))

    return args
