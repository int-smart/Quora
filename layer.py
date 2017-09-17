import tensorflow as tf

def conv_layer(input, filter_shape=[], stride_value=[], padding="VALID", scope_name="conv", activation_fn=tf.nn.relu, k_init=tf.truncated_normal_initializer, b_init=tf.constant_initializer):

    with tf.variable_scope(scope_name) as scope:
        kernel = tf.get_variable(name='weight', shape=filter_shape, initializer=k_init(stddev=0.01))
        biases = tf.get_variable(name='bias', shape=filter_shape[-1], initializer=b_init(0.1))
        conv = tf.nn.conv2d(input, kernel, stride_value, padding=padding)
        conv = tf.nn.bias_add(conv, biases)
        temp = activation_fn(conv)
        return temp

def pool_layer(input, ksize=[], stride=[], padding="VALID",scope_name= "pool"):

    with tf.variable_scope(scope_name) as scope:
        temp = tf.nn.max_pool(input, ksize=ksize, strides=stride, padding=padding)
        return temp

def fc(input, output_channels, scope_name="fc", constant_init=0.1, stddev_init=0.001):
    dim_1, dim_2, dim_3, dim_4 = input.get_shape().as_list()
    #dim_1 = batch_size*number of frames
    input_features = dim_2*dim_3*dim_4
    input = tf.reshape(input, [-1, input_features])
    shape_kernel = [input_features ,output_channels]
#    [-1,input_features]   #reshape this
    with tf.variable_scope(scope_name) as scope:
        kernel = tf.get_variable(name='weight', shape=shape_kernel, initializer=tf.truncated_normal_initializer(stddev=stddev_init))
        biases = tf.get_variable(name='bias', shape=shape_kernel[-1], initializer=tf.constant_initializer(constant_init))
        temp = tf.matmul(input, kernel)+biases
    return temp

if __name__=="__main__":
    pass