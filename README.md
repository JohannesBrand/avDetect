# avDetect
image recognition tool based on tensorflow to detect avalanches in images

## two models:

### self-trained model: not described yet

### inception based model:
to run the model:
- download the model graph from here (too large for github): https://www.dropbox.com/s/xqw8hpghrxpz9to/output_graph.pb?dl=0
- download the labels from here: https://www.dropbox.com/s/esw6jj9raqx7j96/output_labels.txt?dl=0
- install tensorflow and run the runModel.py script which takes 3 params:
  - image_file: path to the image to classify (samples are found in test_pics folder)
  - input_graph: path to model from dropbox
  - input_labels: path to labels from dropbox

(training the inception based model from scratch:
- follow the tensorflow inception retraining tutorial
- when compiling with bazel use the "--copt=-march=native" option)
