# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN][1] on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

![Instance Segmentation Sample][image-1]

The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset


The code is documented and designed to be easy to extend. If you use it in your research, please consider referencing this repository. If you work on 3D vision, you might find our recently released [Matterport3D][2] dataset useful as well.
This dataset was created from 3D-reconstructed spaces captured by our customers who agreed to make them publicly available for academic use. You can see more examples [here][3].

# Projects Using this Model
If you extend this model to other datasets or build projects that use it, we'd love to hear from you.

* [Images to OSM][4]: Use TensorFlow, Bing, and OSM to find features in satellite images.
The goal is to improve OpenStreetMap by adding high quality baseball, soccer, tennis, football, and basketball fields.


# Getting Started
* [demo.ipynb][5] Is the easiest way to start. It shows an example of using a model pre-trained on MS COCO to segment objects in your own images.
It includes code to run object detection and instance segmentation on arbitrary images.

* [train\_shapes.ipynb][6] shows how to train Mask R-CNN on your own dataset. This notebook introduces a toy dataset (Shapes) to demonstrate training on a new dataset.

* ([model.py][7], [utils.py][8], [config.py][9]): These files contain the main Mask RCNN implementation.


* [inspect\_data.ipynb][10]. This notebook visualizes the different pre-processing steps
to prepare the training data.

* [inspect\_model.ipynb][11] This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [inspect\_weights.ipynb][12]
This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns.


# Step by Step Detection
To help with debugging and understanding the model, there are 3 notebooks 
([inspect\_data.ipynb][13], [inspect\_model.ipynb][14],
[inspect\_weights.ipynb][15]) that provide a lot of visualizations and allow running the model step by step to inspect the output at each point. Here are a few examples:



## 1. Anchor sorting and filtering
Visualizes every step of the first stage Region Proposal Network and displays positive and negative anchors along with anchor box refinement.
![][image-2]

## 2. Bounding Box Refinement
This is an example of final detection boxes (dotted lines) and the refinement applied to them (solid lines) in the second stage.
![][image-3]

## 3. Mask Generation
Examples of generated masks. These then get scaled and placed on the image in the right location.

![][image-4]

## 4.Layer activations
Often it's useful to inspect the activations at different layers to look for signs of trouble (all zeros or random noise).

![][image-5]

## 5. Weight Histograms
Another useful debugging tool is to inspect the weight histograms. These are included in the inspect\_weights.ipynb notebook.

![][image-6]

## 6. Logging to TensorBoard
TensorBoard is another great debugging and visualization tool. The model is configured to log losses and save weights at the end of every epoch.

![][image-7]

## 6. Composing the different pieces into a final result

![][image-8]


# Training on MS COCO
We're providing pre-trained weights for MS COCO to make it easier to start. You can
use those weights as a starting point to train your own variation on the network.
Training and evaluation code is in coco.py. You can import this
module in Jupyter notebook (see the provided notebooks for examples) or you
can run it directly from the command line as such:

\`\`\`
# Train a new model starting from pre-trained COCO weights
python3 coco.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 coco.py train --dataset=/path/to/coco/ --model=last
\`\`\`

You can also run the COCO evaluation code with:
\`\`\`
# Run COCO evaluation on the last trained model
python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
\`\`\`

The training schedule, learning rate, and other parameters should be set in coco.py.


# Training on Your Own Dataset
To train the model on your own dataset you'll need to sub-class two classes:

`Config`
This class contains the default configuration. Subclass it and modify the attributes you need to change.

`Dataset`
This class provides a consistent way to work with any dataset. 
It allows you to use new datasets for training without having to change 
the code of the model. It also supports loading multiple datasets at the
same time, which is useful if the objects you want to detect are not 
all available in one dataset. 

The `Dataset` class itself is the base class. To use it, create a new
class that inherits from it and adds functions specific to your dataset.
See the base `Dataset` class in utils.py and examples of extending it in train\_shapes.ipynb and coco.py.

## Differences from the Official Paper
This implementation follows the Mask RCNN paper for the most part, but there are a few cases where we deviated in favor of code simplicity and generalization. These are some of the differences we're aware of. If you encounter other differences, please do let us know.

* **Image Resizing:** To support training multiple images per batch we resize all images to the same size. For example, 1024x1024px on MS COCO. We preserve the aspect ratio, so if an image is not square we pad it with zeros. In the paper the resizing is done such that the smallest side is 800px and the largest is trimmed at 1000px.
* **Bounding Boxes**: Some datasets provide bounding boxes and some provide masks only. To support training on multiple datasets we opted to ignore the bounding boxes that come with the dataset and generate them on the fly instead. We pick the smallest box that encapsulates all the pixels of the mask as the bounding box. This simplifies the implementation and also makes it easy to apply certain image augmentations that would otherwise be really hard to apply to bounding boxes, such as image rotation.

	To validate this approach, we compared our computed bounding boxes to those provided by the COCO dataset.
We found that \~2% of bounding boxes differed by 1px or more, \~0.05% differed by 5px or more, 
and only 0.01% differed by 10px or more.

* **Learning Rate:** The paper uses a learning rate of 0.02, but we found that to be
too high, and often causes the weights to explode, especially when using a small batch
size. It might be related to differences between how Caffe and TensorFlow compute 
gradients (sum vs mean across batches and GPUs). Or, maybe the official model uses gradient
clipping to avoid this issue. We do use gradient clipping, but don't set it too aggressively.
We found that smaller learning rates converge faster anyway so we go with that.

* **Anchor Strides:** The lowest level of the pyramid has a stride of 4px relative to the image, so anchors are created at every 4 pixel intervals. To reduce computation and memory load we adopt an anchor stride of 2, which cuts the number of anchors by 4 and doesn't have a significant effect on accuracy.

## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:
* Speed Improvements. Like re-writing some Python code in TensorFlow or Cython.
* Training on other datasets.
* Accuracy Improvements.
* Visualizations and examples.

You can also [join our team][16] and help us build even more projects like this one.

## Requirements
* Python 3.4+
* TensorFlow 1.3+
* Keras 2.0.8+
* Jupyter Notebook
* Numpy, skimage, scipy, Pillow, cython, h5py

### MS COCO Requirements:
To train or test on MS COCO, you'll also need:
* pycocotools (installation instructions below)
* [MS COCO Dataset][17]
* Download the 5K [minival][18]
  and the 35K [validation-minus-minival][19]
  subsets. More details in the original [Faster R-CNN implementation][20].

If you use Docker, the code has been verified to work on
[this Docker container][21].


## Installation
1. Clone this repository
2. Download pre-trained COCO weights (mask\_rcnn\_coco.h5) from the [releases page][22].
3. (Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

	* Linux: https://github.com/waleedka/coco
	* Windows: https://github.com/philferriere/cocoapi.
	You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

## More Examples
![Sheep][image-9]
![Donuts][image-10]

## Notes about this paper

[1]:	https://arxiv.org/abs/1703.06870
[2]:	https://matterport.com/blog/2017/09/20/announcing-matterport3d-research-dataset/
[3]:	https://matterport.com/gallery/
[4]:	https://github.com/jremillard/images-to-osm
[5]:	/demo.ipynb
[6]:	train_shapes.ipynb
[7]:	model.py
[8]:	utils.py
[9]:	config.py
[10]:	/inspect_data.ipynb
[11]:	/inspect_model.ipynb
[12]:	/inspect_weights.ipynb
[13]:	inspect_data.ipynb
[14]:	inspect_model.ipynb
[15]:	inspect_weights.ipynb
[16]:	https://matterport.com/careers/
[17]:	http://cocodataset.org/#home
[18]:	https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0
[19]:	https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0
[20]:	https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md
[21]:	https://hub.docker.com/r/waleedka/modern-deep-learning/
[22]:	https://github.com/matterport/Mask_RCNN/releases

[image-1]:	assets/street.png
[image-2]:	assets/detection_anchors.png
[image-3]:	assets/detection_refinement.png
[image-4]:	assets/detection_masks.png
[image-5]:	assets/detection_activations.png
[image-6]:	assets/detection_histograms.png
[image-7]:	assets/detection_tensorboard.png
[image-8]:	assets/detection_final.png
[image-9]:	assets/sheep.png
[image-10]:	assets/donuts.png