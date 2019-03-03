import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def expand(x, dim, N):
    return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)

def learned_init(units):
    """
        输入是一个标量1, 通过一个全连接层将其映射为 units 个神经元
        输出将作为 NTM 存储单元的值. tf.squeeze(shape=(1,units)) = shape=(units,). 
        输出为长度为 units 的向量
    """
    return tf.squeeze(tf.contrib.layers.fully_connected(tf.ones([1, 1]), units,
        activation_fn=None, biases_initializer=None))

def create_linear_initializer(input_size, dtype=tf.float32):
    stddev = 1.0 / np.sqrt(input_size)
    return tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)

def plot_heatmap(n, inputs, outputs):
    figures=[("Inputs", inputs[0].T),  ('Outputs', outputs[0].T)]

    fig, axeslist = plt.subplots(ncols=2, nrows=1, gridspec_kw={'width_ratios': [3,1]})

    for ind, (title, fig) in enumerate(figures):
        axeslist.ravel()[ind].imshow(fig, cmap='gray', interpolation='nearest')
        axeslist.ravel()[ind].set_title(title)
        if ind == 0:
            axeslist.ravel()[ind].set_xlabel('Time ------->')
    plt.sca(axeslist[1])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("save/epoch_"+str(n)+".jpg")
    