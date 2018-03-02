
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import time

'''
import h5py
dataset = h5py.File('C:\\Users\\45969\OneDrive\Document\TensorFlow\VQA\data.h5','r')
train_x = dataset['cifar10_X']
train_y = dataset['cifar10_Y']
test_x = dataset['cifar10_X_test']
test_y = dataset['cifar10_Y_test']
train_y = np.argmax(train_y, axis=1)
test_y = np.argmax(test_y, axis=1)
'''
import pickle
with open('cifar_10.pkl', 'rb') as f:
    data = pickle.load(f)
train_x, train_y, test_x, test_y =             data['train_x'],data['train_y'],data['test_x'],data['test_y']
    
def conv_layers(net_in):
    #'''
    with tf.name_scope('process') as scope:
        mean = tf.constant([123.68, 116.779, 103.93], dtype=tf.float32,
                           shape=[1,1,1,3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
        #'''
    network = Conv2d(net_in, 64,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu,name='conv1_1')
    network = Conv2d(network, 64,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name='conv1_2')
    network = MaxPool2d(network, (2,2),(2,2),padding='SAME', name ='pool1')
    network = Conv2d(network, 128,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv2_1')
    network = Conv2d(network, 128,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv2_2')
    network = MaxPool2d(network, (2,2),(2,2),padding='SAME', name ='pool2')
    network = Conv2d(network,256,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv3_1')
    network = Conv2d(network, 256,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv3_2')
    network = Conv2d(network, 256,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv3_3')
    network = MaxPool2d(network, (2,2),(2,2),padding='SAME', name ='pool3')
    network = Conv2d(network, 512,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv4_1')
    network = Conv2d(network, 512,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv4_2')
    network = Conv2d(network, 512,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv4_3')
    network = MaxPool2d(network, (2,2),(2,2),padding='SAME', name ='pool4')
    """ conv5 """
    #'''
    network = Conv2d(network, 512,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv5_1')
    network = Conv2d(network, 512,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv5_2')
    network = Conv2d(network, 512,(3,3),(1,1), padding='SAME',
                      act=tf.nn.relu, name ='conv5_3')
    network = MaxPool2d(network, (2,2),(2,2),padding='SAME', name ='pool5')
    #'''
    return network

def fc_layers(net, is_train=True):
    network = FlattenLayer(net, name='flatten')
    network = DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc1_relu')
    network = DropoutLayer(network, keep=0.5, is_train=is_train,name='drop1')
    network = DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc2_relu')
    last2 = network.outputs
    network = DropoutLayer(network, keep=0.5, is_train=is_train,name='drop2')
    network = DenseLayer(network, n_units=10, act=tf.identity, name='fc3_relu')
    return network

def distort_fn(x, is_train=False):
    """
    Description
    -----------
    The images are processed as follows:
    .. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
    .. They are approximately whitened to make the model insensitive to dynamic range.
    For training, we additionally apply a series of random distortions to
    artificially increase the data set size:
    .. Randomly flip the image from left to right.
    .. Randomly distort the image brightness.
    """
    # print('begin',x.shape, np.min(x), np.max(x))
    #x = tl.prepro.crop(x, 24, 24, is_random=is_train)
    # print('after crop',x.shape, np.min(x), np.max(x))
    if is_train:
        # x = tl.prepro.zoom(x, zoom_range=(0.9, 1.0), is_random=True)
        # print('after zoom', x.shape, np.min(x), np.max(x))
        x = tl.prepro.flip_axis(x, axis=1, is_random=True)
        # print('after flip',x.shape, np.min(x), np.max(x))
        x = tl.prepro.brightness(x, gamma=0.1, gain=1, is_random=True)
        # print('after brightness',x.shape, np.min(x), np.max(x))
        # tmp = np.max(x)
        # x += np.random.uniform(-20, 20)
        # x /= tmp
    # normalize the image
    x = (x - np.mean(x)) / max(np.std(x), 1e-5) # avoid values divided by 0
    # print('after norm', x.shape, np.min(x), np.max(x), np.mean(x))
    return x


tf.reset_default_graph()
sess = tf.InteractiveSession()

tl.layers.clear_layers_name()

x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='image')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='label')

net_in = InputLayer(x, name='input')
net_cnn = conv_layers(net_in)
network = fc_layers(net_cnn)


y = network.outputs


probs = tf.nn.softmax(y)
y_op = tf.argmax(tf.nn.softmax(y), 1)
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=y_, name='cost'))
yyy = tf.argmax(y,1)

# only regularize fully conncect layer
#'''
L2 = 0
for w in tl.layers.get_variables_with_name('relu/W', True, False):
    L2 += tf.contrib.layers.l2_regularizer(0.004)(w)
cost = cost+L2
#'''

correct_prediction = tf.equal(tf.cast(tf.argmax(y,1),tf.float32), tf.cast(y_,tf.float32))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learning_rate = 0.0001
train_params = network.all_params[20:]
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9,
                                  beta2=0.999, epsilon=1e-08, 
                                  use_locking=False).minimize(cost, var_list=train_params)

tl.layers.initialize_global_variables(sess)

weights = np.load('C:\\Users\\45969\OneDrive\Document\TensorFlow\VQA\data\\vgg16_weight\\vgg16_weights.zip')
param = []
for key in sorted(weights.keys()):
    param.append(weights[key])
tl.files.assign_params(sess, param[:20], network)

network.print_params()
network.print_layers()



# In[ ]:


n_epoch = 500
batch_size = 256

print_freq = 1


for epoch in range(n_epoch):
    for train_x_a, train_y_a in tl.iterate.minibatches(train_x,
                                                       train_y, batch_size):
        #train_x_a = tl.prepro.threading_data(train_x_a, fn=distort_fn, is_train=True)
        feed_dict={x:train_x_a, y_:train_y_a}
        feed_dict.update(network.all_drop)
        _, pred = sess.run([train_op, y_op], feed_dict=feed_dict)

    if epoch + 1 == 1 or (epoch+1) % print_freq == 0:
        print('Epoch %d of %d' % (epoch+1, n_epoch))
        test_loss, test_acc, n_batch = 0,0,0
        for train_x_a, train_y_a in tl.iterate.minibatches(test_x,
                                                       test_y, batch_size):
            #train_x_a = tl.prepro.threading_data(train_x_a, fn=distort_fn, is_train=False)
            dp_dict = tl.utils.dict_to_one(network.all_drop)
            feed_dict={x:train_x_a, y_:train_y_a}
            feed_dict.update(network.all_drop)
            err, ac = sess.run([cost, acc],feed_dict=feed_dict)
            test_loss += err
            test_acc += ac
            n_batch += 1
        print('Save model, data' + '!'*10)
        tl.files.save_npz_dict(network.all_params, name='cifar_tune.npz', sess=sess)
        print('test loss: %f' % (test_loss/n_batch))
        print('test acc: %f' % (test_acc/n_batch))


# In[ ]:


# accuracy rate doesn't improve until epoch 10
network.print_params()

