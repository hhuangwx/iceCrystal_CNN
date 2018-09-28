import tensorflow as tf
import sys
import numpy
import math

# Number of classes
nClass=8

# Dimensions of image (pixels)
height=120
width=120

# Function to tell TensorFlow how to read a single image from input file
def getImage(filename):
    # convert filenames to a queue for an input pipeline.
    filenameQ = tf.train.string_input_producer([filename],num_epochs=None)

    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example
    key, fullExample = recordReader.read(filenameQ)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/channels':  tf.FixedLenFeature([], tf.int64),
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })


    # now we are going to manipulate the label and image features
    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg',[image_buffer], None):
        # decode image from jpeg into array
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    #greyscale the image
    # the "1-.." part inverts the image, so that the background is black.
    image=tf.reshape(1-tf.image.rgb_to_grayscale(image),[height*width])

    # re-define label as a "one-hot" vector
    # i.e. [0,0,0,0,0,0,0,1]
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, nClass))

    return label, image


# associate the "label" and "image" objects with the corresponding features read from
# a single example in the training data file
label, image = getImage("data/train-00000-of-00001")

# and similarly for the validation data
vlabel, vimage = getImage("data/validation-00000-of-00001")

# associate the "label_batch" and "image_batch" objects with a randomly selected batch---
# of labels and images respectively
imageBatch, labelBatch = tf.train.shuffle_batch(
    [image, label], batch_size=100,
    capacity=2000,
    min_after_dequeue=1000)

# and similarly for the validation data
vimageBatch, vlabelBatch = tf.train.shuffle_batch(
    [vimage, vlabel], batch_size=429,
    capacity=2000,
    min_after_dequeue=1000)

# interactive session allows inteleaving of building and running steps
sess = tf.InteractiveSession()

# x is the input array, which will contain the data from an image
# this creates a placeholder for x, to be populated later
x  = tf.placeholder(tf.float32, [None, width*height])
# similarly, we have a placeholder for true outputs (obtained from labels)
y_ = tf.placeholder(tf.float32, [None, nClass])

# run convolutional neural network model given in "Expert MNIST" TensorFlow tutorial

# functions to init small positive weights and biases
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# set up "vanilla" versions of convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                      strides=[1, 2, 2, 1], padding='SAME')
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                      strides=[1, 3, 3, 1], padding='SAME')

print("Running Convolutional Neural Network Model")
nFeatures1=32
nFeatures2=64
nNeuronsfc=1024

# use functions to init weights and biases
# nFeatures1 features for each patch of size 5x5
# SAME weights used for all patches
# 1 input channel
W_conv1 = weight_variable([5, 5, 1, nFeatures1])
b_conv1 = bias_variable([nFeatures1])

# reshape raw image data to 4D tensor. 2nd and 3rd indexes are W,H, fourth
# means 1 colour channel per pixel
# x_image = tf.reshape(x, [-1,28,28,1])
x_image = tf.reshape(x, [-1,width,height,1])


# hidden layer 1
# pool(convolution(Wx)+b)
# pool reduces each dim by factor of 2.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# similarly for second layer, with nFeatures2 features per 5x5 patch
# input is nFeatures1 (number of features output from previous layer)
W_conv2 = weight_variable([5, 5, nFeatures1, nFeatures2])
b_conv2 = bias_variable([nFeatures2])


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
################################################################################
#These are extra convolution/maxpooling layers to test out affect on network.
nFeatures3 = 40
nFeatures4 = 40
nFeatures5 = 40
nFeatures6 = 40
nFeatures7 = 40

W_conv3 = weight_variable([3, 3, nFeatures2, nFeatures3])
b_conv3 = bias_variable([nFeatures3])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

# #layer 4
# W_conv4 = weight_variable([5, 5, nFeatures3, nFeatures4])
# b_conv4 = bias_variable([nFeatures4])
# h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
# h_pool4 = max_pool_3x3(h_conv4)
# print(tf.shape(h_pool4))
#
# #layer 5
# W_conv5 = weight_variable([5, 5, nFeatures4, nFeatures5])
# b_conv5 = bias_variable([nFeatures5])
# h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
# h_pool5 = max_pool_2x2(h_conv5)
# print(tf.shape(h_pool5))
#
# #layer 6
# W_conv6 = weight_variable([5, 5, nFeatures5, nFeatures6])
# b_conv6 = bias_variable([nFeatures6])
# h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
# h_pool6 = max_pool_2x2(h_conv6)
# #layer 7
# W_conv7 = weight_variable([5, 5, nFeatures6, nFeatures7])
# b_conv7 = bias_variable([nFeatures7])
# h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
# h_pool7 = max_pool_2x2(h_conv7)

################################################################################
numlayers= 3 #+math.log(3,2)

# check our dimensions are a multiple of 4
if (width%(2**numlayers) or height%(2**numlayers)):
    print("Size Error")
    sys.exit(1)

W_fc1 = weight_variable([int(int(width)/(2**numlayers)) * int(int(height)/(2**numlayers)) * int(nFeatures3), int(nNeuronsfc)])
b_fc1 = bias_variable([nNeuronsfc])

# flatten output from previous layer
h_pool2_flat = tf.reshape(h_pool3, [-1, int(int(width)/(2**numlayers)) * int(int(height)/(2**numlayers)) * int(nFeatures3)])
h_fc1        = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# reduce overfitting by applying dropout
# each neuron is kept with probability keep_prob
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# create readout layer which outputs to nClass categories
W_fc2 = weight_variable([nNeuronsfc, nClass])
b_fc2 = bias_variable([nClass])

# define output calc (for each class) y = softmax(Wx+b)
# softmax gives probability distribution across all classes
# this is not run until later
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#create instance to save weights and biasies into
saver = tf.train.Saver()

#define loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+1e-30), reduction_indices=[1]))

#define training step which minimises cross entropy
train_step    = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#argmax gives index of highest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#get mean of all entries in correct prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#run the session #############

#initialize the variables
sess.run(tf.global_variables_initializer())

#start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

#start training
nSteps=10
maxTA = 0
for i in range(nSteps):
    probabilities      = y
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    #print(tf.Tensor.get_shape(batch_xs))

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    #perform validation
    if (i+1)%1 == 0:
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        train_accuracy       = accuracy.eval(feed_dict={x:vbatch_xs, y_:vbatch_ys, keep_prob: 0.9})

        if train_accuracy > maxTA:
          maxTA = train_accuracy

        print("step %d, TA %g"%(i+1, train_accuracy))
        print("maxTA:", maxTA)

      #print("probabilities", probabilities.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob:1.0}, session=sess))


saver.save(sess, './saved_model/ice_crystal_Conv')

# finalise
coord.request_stop()
coord.join(threads)
