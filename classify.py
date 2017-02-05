import tensorflow as tf
import numpy as np
import sys
import glob, os
from PIL import Image

image_size = 64  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
num_channels = 1
batch_size = 16
num_labels = 2

# modelDir = 'model/00000001/'
modelDir = 'model/'
#image = sys.argv[1] #'../Pictures/avalanche/Schneebrettlawine/ActiOn_9.thumbnail'
rDir = "../Pictures/avalanche/"
dirs = os.listdir(rDir)
image = rDir + dirs[1] + "/ActiOn_56.thumbnail"
label = 0
img = Image.open(image).convert("L")
image_data = (np.asarray(img).astype(np.float32)-pixel_depth / 2) / pixel_depth
if image_data.shape != (image_size, image_size):
    raise Exception('Unexpected image shape: %s' % str(image_data.shape))

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels).astype(np.float32).reshape(-1,num_labels)
  return dataset, labels

image_data, labels = reformat(image_data,label)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def isAvalanche(predictions,labels):
    return (np.argmax(predictions, 1) == np.argmax(labels, 1))

sess = tf.Session()
saver = tf.train.import_meta_graph(modelDir + 'my_model.meta')
saver.restore(sess, modelDir + 'my_model') #tf.train.latest_checkpoint(modelDir))
#model1 = tf.get_collection('model')[0]
tf_class_dataset = tf.get_collection('tf_class_dataset')[0]
tf_class_labels = tf.get_collection('tf_class_labels')[0]
class_prediction = tf.get_collection('class_prediction')[0]
feed_dict = {tf_class_dataset: image_data, tf_class_labels: labels}
predictions = sess.run(class_prediction, feed_dict=feed_dict)
print(predictions)
if isAvalanche(predictions,labels):
    print("Classified as avalanche")
else:
    print("Classified as no avalanche")
sess.close()

# feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
#feed_dict = {tf_train_dataset: image_data, tf_train_labels: label}
#_, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
#print(image_data.shape)
# savedStuff = tf.get_collection('layer1_weights')
# print(len(savedStuff))
# pred = sess.run(tf.convert_to_tensor(image_data))
# print(pred.shape)
#print('Is avalanche: %d%%' % isAvalanche(pred, label))
#print(isAvalanche(pred,label).shape)
#sess.close()
      #[optimizer, loss, train_prediction], feed_dict=feed_dict)
  #pred = sess.run(batch_data)
  #test_prediction = tf.nn.softmax(model(batch_data))
  #tf.get_default_session().run(test_prediction)

  #print(test_prediction.eval(session=sess2))
  #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), batch_labels))
  #feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
  #conv = tf.nn.conv2d(image_data, layer1_weights, [1, 1, 1, 1], padding='SAME')
  #print(layer1_weights)
  # tf_image = tf.constant(image_data)
  # tf_label = tf.constant(1)
  #feed_dict = {tf_train_dataset: image_data, tf_train_labels: label}
  #pred = sess.run(image_data)
  # print(pred.shape)


  # def accuracy(predictions, labels):
  #     return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
  #             / predictions.shape[0])

  # tf.get_collection() returns a list. In this example we only want the
  # first one.
  #train_op = tf.get_collection('train_op')[0]
  #for step in xrange(1000000):
  #  sess.run(train_op)