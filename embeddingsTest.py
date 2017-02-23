import numpy as np
import tensorflow as tf
import os
from six.moves import cPickle as pickle
from six.moves import range
from tensorflow.contrib.tensorboard.plugins import projector
#from tensorflow.contrib.session_bundle import exporter

# tf.app.flags.DEFINE_integer('training_iteration', 1000,
#                             'number of training iterations.')
# tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
# tf.app.flags.DEFINE_string('work_dir', './', 'Working directory.')
# FLAGS = tf.app.flags.FLAGS

logs_path = 'emb_logs/'
pickle_file = 'avDetect.pickle'
export_path = logs_path #"model/"

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  org_train_dataset = save['train_dataset']
  org_train_labels = save['train_labels']
  org_test_dataset = save['test_dataset']
  org_test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', org_train_dataset.shape, org_train_labels.shape)
  print('Test set', org_test_dataset.shape, org_test_labels.shape)

image_size = org_train_dataset.shape[1]
num_labels = 2
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(org_train_dataset, org_train_labels)
test_dataset, test_labels = reformat(org_test_dataset, org_test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

def save_metadata(file,labels):
    with open(file, 'w') as f:
        for l in labels:
            f.write(str(l)+"\n")
save_metadata(os.path.join(logs_path, 'metadata.tsv'),org_train_labels)

# actual code for embeddings
sess = tf.InteractiveSession()

embedding_var = tf.Variable(tf.stack(org_train_dataset, axis=0), trainable=False, name='embedding_var')

tf.global_variables_initializer().run()

saver = tf.train.Saver(write_version=1)
writer = tf.summary.FileWriter(logs_path, sess.graph)

# Add embedding tensorboard visualization. Need tensorflow version
# >= 0.12.0RC0
config = projector.ProjectorConfig()
embed= config.embeddings.add()
embed.tensor_name = embedding_var.name
embed.metadata_path = os.path.join(logs_path,'metadata.tsv')

projector.visualize_embeddings(writer, config)

saver.save(sess, os.path.join(logs_path, 'a_model.ckpt'))




