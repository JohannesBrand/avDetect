import matplotlib.pyplot as plt
import numpy as np
import glob, os
from six.moves import cPickle as pickle
from PIL import Image

image_size = 64  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

pTestData = 0.2 # percent of data used for test

picturePath = "../Pictures/"
classFolders = os.listdir(picturePath)

def load_folder(folder, min_num_images):
    """Load the data for one folder."""
    image_files = glob.glob(folder + "*.thumbnail")
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    num_images = 0
    for image in image_files:
        try:
            img = Image.open(image).convert("L")
            image_data = (np.asarray(img).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def class_data(class_folder, min_num_images_per_class, force=False):
    set_filename = class_folder + '.pickle'
    data_folders = os.listdir(class_folder)

    if os.path.exists(set_filename) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping pickling.' % set_filename)
    else:
        print('Pickling %s.' % set_filename)
        dataset = []
        for folder in data_folders:
            dataset.append(load_folder(class_folder + "/" + folder + "/", min_num_images_per_class))
        try:
            with open(set_filename, 'wb') as f:
                pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', set_filename, ':', e)
    return set_filename


class1 = class_data(picturePath + classFolders[0], 10)  # 45000
class2 = class_data(picturePath + classFolders[1], 10)  # 45000

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def combine_data(class_datafiles,pTest=0.2):
    train_dataset = np.empty([0, image_size, image_size], dtype=np.float32)
    train_labels = np.empty(shape=0, dtype=np.int32)
    test_dataset = np.empty([0, image_size, image_size], dtype=np.float32)
    test_labels = np.empty(shape=0, dtype=np.int32)
    l = 0
    for c in class_datafiles:
        dataset = np.empty([0, image_size, image_size], dtype=np.float32)
        labels = np.empty(shape=0, dtype=np.int32)
        with open(c, 'rb') as f:
            data = pickle.load(f)
            for d in data:
                dataset = np.vstack((dataset, d))
                labels = np.hstack((labels, np.tile(l, len(d))))

        l += 1
        dataset, labels = randomize(dataset,labels)
        splitindex = int(pTest*dataset.shape[0])
        train_dataset = np.vstack((train_dataset,dataset[splitindex:,:,:]))
        train_labels = np.hstack((train_labels, labels[splitindex:]))
        test_dataset = np.vstack((test_dataset, dataset[0:splitindex, :, :]))
        test_labels = np.hstack((test_labels, labels[0:splitindex]))

    return train_dataset, train_labels, test_dataset, test_labels  # , np.ndarray(labels,dtype=np.int32)

# dataset, labels =
train_dataset, train_labels, test_dataset, test_labels = combine_data([class1, class2],pTestData)
train_dataset, train_labels = randomize(train_dataset,train_labels)
test_dataset, test_labels = randomize(test_dataset,test_labels)

pickle_file = 'avDetect.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

#print(train_dataset.shape[0])
#print(test_dataset.shape[0])

#plt.figure()
#plt.imshow(train_dataset[2,:,:],cmap="jet")
#plt.show()

#print(train_dataset[2,:,:].min())
#print(train_dataset[2,:,:].max())
