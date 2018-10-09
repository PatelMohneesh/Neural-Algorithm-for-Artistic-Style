import numpy as np
import tensorflow as tf

def content_loss(cont_out, target_out, layer, content_weight):
    cont_loss = tf.reduce_sum(tf.square(tf.subtract(target_out[layer], cont_out)))
    cont_loss = tf.multiply(cont_loss, content_weight, name="cont_loss")
    return cont_loss

def get_shape(inp):
    if type(inp) == type(np.array([])):
        return inp.shape
    else:
        return [i.value for i in inp.get_shape()]

def style_loss(style_out, target_out, layers, style_weight_layer):

    def style_layer_loss(style_out, target_out, layer):
        def gram_matrix(activation):
            flat = tf.reshape(activation, [-1, get_shape(activation)[3]]) # shape[3] is the number of feature maps
            res = tf.matmul(flat, flat, transpose_a=True)
            return res

        N = get_shape(target_out[layer])[3] # number of feature maps
        M = get_shape(target_out[layer])[1] * get_shape(target_out[layer])[2] # dimension of each feature map
        
        style_gram = gram_matrix(style_out[layer])
        target_gram = gram_matrix(target_out[layer])

        st_loss = tf.multiply(tf.reduce_sum(tf.square(tf.subtract(target_gram, style_gram))), 1./((N**2) * (M**2)))

        st_loss = tf.multiply(st_loss, style_weight_layer, name='style_loss')

        return st_loss

    losses = []
    for s_l in layers:
        loss = style_layer_loss(style_out, target_out, s_l)
        losses.append(loss)

    return losses

def total_var_loss(generated, tv_weight):
    batch, width, height, channels = get_shape(generated)

    width_var = tf.nn.l2_loss(tf.subtract(generated[:,:width-1,:,:], generated[:,1:,:,:]))
    height_var = tf.nn.l2_loss(tf.subtract(generated[:,:,:height-1,:], generated[:,:,1:,:]))

    return tv_weight*tf.add(width_var, height_var)