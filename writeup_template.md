# Traffic Sign Recognition

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/sign_1.jpg "Traffic Sign 1"
[image5]: ./test_images/sign_2.jpg "Traffic Sign 2"
[image6]: ./test_images/sign_3.jpg "Traffic Sign 3"
[image7]: ./test_images/sign_4.jpg "Traffic Sign 4"
[image8]: ./test_images/sign_5.jpg "Traffic Sign 5"
[image9]: ./examples/obs_per_class.png "Observations per Class"
[image10]: ./examples/before_grayscale.png "Before Grayscale"
[image11]: ./examples/before_grayscale_bounded.png "Before Grayscale - Bounded"
[image12]: ./examples/after_grayscale_and_roi_masking.png "After Grayscale and Region of Interest masking"
[image13]: ./examples/affine_transform.png "Affine Transform"
[image14]: ./examples/obs_per_class_after_augmentation.png "Observations per Class"
[image15]: ./examples/train_loss_vs_validation_loss.png "Loss Curves"
[image16]: ./examples/train_accuracy_vs_validation_accuracy.png "Accuracy"
[image17]: ./examples/predictions.png "Predictions"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it, and here is a [link to my project code](https://github.com/adriantorrie/udacity_sdcnd_project_2_traffic_sign_classifier/Traffic_Sign_Classifier.ipynb), and the [`helpers.py`](https://github.com/adriantorrie/udacity_sdcnd_project_2_traffic_sign_classifier/helpers.py) script I wrote to clean things up a bit.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the python methods such as `len()` to determine counts, and `set()` to find unique values. The `nupmy.array.shape` property was used to read the dimensions of the sign images in the data set:

```
    Number of training examples     = 34,799
    Number of validation examples   =  4,410
    Number of testing examples      = 12,630
    ----------------------------------------
    Total examples                  = 51,839
    
    Image data shape    = (32, 32, 3)
    Number of classes   = 43
```

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across classes. As is easily discernible, some images have a higher number included. Those images with higher counts/frequencies will cause the neural network to more likely memorise their features, and therefore less likely to generalise. This will reduce predictive power if not accounted for (which is described later).

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. 

* What techniques were chosen and why did you choose these techniques?
* Consider including images showing the output of each preprocessing technique.
* Pre-processing refers to techniques such as 
  * converting to grayscale
  * normalization, etc. 

(OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because
 * I wanted to reduce the amount of data going through the network for processing to speed things up
 * and because [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) touted the benefits of grayscaling:

>**"...produced a new record of 99.17% by ... and by using greyscale images instead of color."**

Here is an example of an unprocessed traffic sign image 

![alt text][image10]

and after finding the boundary box (which will be used to find the Region of Interest)

![alt text][image11]

After grayscaling and masking for the Region of Interest

![alt text][image12]

As a last step, I normalized the image data because it improves the search space for gradient descent. Below is an example (from Andrej Karpathy's [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/) course) that shows how normalising provides a more centric data set. this can make gradient descent converge faster.

![Alt](https://cs231n.github.io/assets/nn2/prepro1.jpeg "Normalised dimensions")

I decided to generate additional data because the data was skewed towards some classes. So I wanted to provide more examples for those image that were under-represented. This would allow the model to generalise more than memorise.

To add more data to the the data set, I used the following techniques:
* affine transformation
  * this provided 'alternative' images
  * not all images were front on, 'clean' images of the sign
  * using the affine transform essentially provided a view of the image from another perspective/angle.
  * it's like taking multiple photos of the same sign from different angles so the model can recognise it no matter what angle the sign is on.
* rotated images
  * not all images were front on, 'clean' images of the sign.
  * similar reasons for affine transform
* region of interest images
  * while I wanted the model to generalise, I also wanted to ensure it learnt the important features that made up a sign by removing background noise that wasn't necessary for determining sign type.

Here is an example of an affine transform of a grayscale image, that has the region of interest mask applied to it:

![alt text][image13]

The difference between the original data set and the augmented data set is the following:
 * number of images across classes is more uniform, in an effort to prevent the model from overfitting towards the most common images in the original set
 * higher number of images, that provide a view of the image from a different perspective, to allow the model to generalise better

![alt text][image14]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					     | 
|:---------------------:|:--------------------------------------------------:| 
| Input         		| 32x32x1 grayscale image + 32x32x1 augmented images | 
| Convolution 1x1     	| 1x1 stride, same padding	     |
|   ELU activation		| [Better than RELU](https://arxiv.org/pdf/1511.07289.pdf)  								 |
|   Max pooling	      	| 2x2 stride 				     |
| Convolution 1x1	    | etc.      									     |
|   ELU activation      | [Better than RELU](https://arxiv.org/pdf/1511.07289.pdf)                                   |
|   Max pooling         | 2x2 stride                     |
| Inception module      | 1x1 + 3x3 + 5x5 + avg pooling                      |
|   ELU activation      | [Better than RELU](https://arxiv.org/pdf/1511.07289.pdf) 
| Flattened layer       | Conv1 + Conv2 (Sermanet) + Inception Module        |
| Fully connected		| 120 hidden nodes  								 |
| Fully connected       | 84 hidden nodes                                    |
| Outputs (logits)      | 43 hidden nodes (1 for each class)                 |
| Softmax				| function: softmax(logits) used to make predictions |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

```
# batching
epochs = 20
batch_size = 128
in_channels = 1
inception_filters = 1

# weight initialisation
mu = 0
sigma = 0.1

# dropout
keep_prob = 0.75

# optimiser
starter_learning_rate = 1e-3
decay_steps = len(X_train) / 4
decay_rate = 0.85
```

The greatest accuracy improvement impact on the validation set was caused by two things:
* grayscaling images
* adjusting the learning rate
  * using an adaptive learning rate reduced the 'noise' in the loss function in later epochs during training.

Optimsers tested were:
* [tf.train.AdagradOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
* [tf.train.AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)

`tf.train.AdamOptimizer` gave superior results.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
```
training set accuracy of:   0.798 (best at epoch 20)
validation set accuracy of: 0.940 (best at epoch 14)
test set accuracy of:       0.901
```

Loss and accuracy curves are shown below:
![alt text][image15]
![alt text][image16]

* LeNet was used to do the basics, grayscaling, normalisation, learning rate. This got high 80's validation accuracy
* Sermanet was then used, this pushed validation into low-mid 90's
* Modifying the images to have the region of interest masked was then performed, minor improvement was made.
* Epochs were pushed up to 200. No improvement seen after 20, however validation accuracy got over 95% at one stage.
* Epochs dropped to 20, seemed little was being gained.
* Inception module added just for fun to see if it helped. Didn't seem too.
* Inception network with one module and 2 fully connect layers was tested, high 80's acheived for accuracy without any further tuning, so went settled on the LeNet + Sermanet + Inception Module combination instead.
* Inception module left in for the same reason the Sermanet variation was performed, to allow lower level feature generalisations to mix with higher level feature generalisations.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  * LeNet
  * It was in a prior lab and I was familiar with it
* What were some problems with the initial architecture?
  * Low accuracy when run vanilla with colour (RGB - 3 channel) images
* How was the architecture adjusted and why was it adjusted?
  * Added ELU activations instead of RELU.
  * Sermanet variation (easy to experiment with once reading the paper, and it worked well, so I kept it)
  * Inception module added, wanted ability to generalise (and kinda cool to be learning more about it)
* Which parameters were tuned? How were they adjusted and why?
  * Epochs
    * When learning rate was made smaller, I allowed more epochs as the intial epochs don't move as quickly.
    * When learning rate was adjusted to be larger I lowered the epochs
    * Sometimes I lowered epochs when playing with different architectures (usually to 10) to see how quickly the validation accuracy would improve.
  * Learning rate
    * Wanted the loss to be as smoothly descending as possible.
    * Sometimes it was too high for any architecture changes I made, other times too low.
  * Batch size
    * Memory was looked at to try to be using as much of the computer resources as possible for performance.
    * `top` was used to monitor system resource during training
    * [GPU memory was looked at](https://unix.stackexchange.com/a/38581/29794) to maximise utilisation for performance reasons
    * `watch -n 0.5 nvidia-smi` was used to monitor GPU during training
  * Number of inception filters
    * I used a large number of filters at one point (~256), was too slow to iterate over while sitting there watching. Settled on 1 purely for performance reasons, I was more interested in getting it too work so that I understood the architecture of an inception module from a code perspective.
  * Dropout rate
    * Was lowered when I made epochs higher. I left it at 0.75, If I had run my model for higher number of epochs I would've set this higher to prevent over-fitting. I made the trade-off between overfitting/performance in an effort to complete the project.
  * Weight initialisation wasn't changed, and was left at `mu = 0`, and `sigma = 0.1`.
    * The [current recommended weight initialisation method](https://cs231n.github.io/neural-networks-2/#init) (using `ReLu`) is `w = np.random.randn(n) * sqrt(2.0/n)`. Because I used the `Elu` instead I left the initialisation using the uniform distribution with `mu` and `sigma`, for symmetry breaking, as I couldn't find any information that `Elu` variance would be so different from zero (unlike `Relu`s which have been shown to have variance greater than one, and therefore why the optimised initilisation mentioned is suited to `Relu`s).
    * The Elu paper states
    * >ELUs have negative
values which allows them to push **mean unit activations closer to zero** like batch
normalization but with lower computational complexity
    * and ...
    * > **Mean shifts toward zero**
speed up learning by bringing the normal gradient closer to the unit natural gradient
because of a reduced bias shift effect.
    * This is why I decided to stick with weight initialization of `mu = 0` and `sigma = 0.1`
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  * CNNs work well for computer vision problems because of their ability to learn low, medium, and high level features (when using a 3 layer CNN with each layer feeding into the next). Given signs aren't complex shapes, like human faces, and are fairly uniform in their appearance, only the low and medium level features should be needed to be learned in order to pass the threshold for this project (93% validation accuracy).
    * This is why two convolution layers were kept, instead of three
    * The Sermanent variation of passing the first convolution layer's features through to the fully connected layer made improvements significant enough, and the paper explaining reusing features the network had already learnt made intuitive sense to me.
  * Dropout was only used on the fully connected layers. It degraded performance substantially when applying dropout to convolution layers, and the layers within the inception module. Higher number epochs, with higher learning rates would've been needed to train with dropout factored into each layer.

If a well known architecture was chosen:
* What architecture was chosen?
  * LeNet + Sermanet variation
* Why did you believe it would be relevant to the traffic sign application?
  * The paper that discussed the architecture was specifically for road signs
  * Preprocessing of images was discussed, along with possible improvements via data augmentation.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  * The accuracies of the three sets all confirm one another. 
  * By this, we mean to look at how the accuracies move with respect to one another through the training process (train + validation accuracy and loss). Both had smooth curves and didn't deviate, or act in a noisy fashion.
  * The test accuracy confirmed the validation accuracy, as it was in the same ballpark, which would suggest the model wasn't overfitting.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. 

*For each image, discuss what quality or qualities might be difficult to classify.*

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images are scaled down to 32x32, before being converted to grayscale images.

* The first image might be difficult to classify because:
  * It doesn't exist in the dataset, so it shouldn't be able to classify something it hasn't 'seen' before.
* The second image might be difficult to classify because:
  * The image is longer in height than width, so when resizing to a 32x32 image I'd expect some of the curves of the numbers to be inconsistent with what the model has learned, as the '60' should be malformed by the scaling of the image.
* The third image might be difficult to classify because:
  * I'm not expecting any issues with classifying his sign. The image is well lit, the background is well contrasted with the sign itself, and the shape of the image will hold up well to rescaling down to 32x32.
* The fourth image might be difficult to classify because:
  * Not expecting any issues with this one, as per the notes with image 3.
* The fifth image might be difficult to classify because:
  * Like image 2 (60 km/h speed limit) it is longer in height. However, the sign itself is well centered , so any down-scaling should have less of an impact with regards to warping any edges/curves that are to be detected as features.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

*At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).*

Here are the results of the prediction:

![alt text][image17]

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. However, I purposefully tried predicting against an unseen image for curiosity of what it would predict as much as any thing else, therefore it could be said the model correctly predicted 3 out of 4 images, giving an accuracy of 75%. This compares favorably to the accuracy on the test set of 79%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.
*Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)*

For the first image, the model is relatively sure that this is a Double curve sign (probability of 0.8871), and the image does contain a stop sign. The top five soft max probabilities can be seen in the image above, in the horizontal bar plot next to each image the prediction was made for.

For the second image its top 5 predictions were all speed signs, however it was unable to classify the correct speed limit. It was relatively split over whether it was  50km/h or 20km/h, even though neither were true. Interestingly, the 60km/h sign was not in the top 5 predictions what-so-ever. I thought the shape of the scaled image would affect it's softmax outcome, but not to the point where it wold incorrectly classify it, nor did I exepect that it wouldn't even be one of the top 5 choices all together. Further investigation into the architecture of successful MINST classifiers may provide some insight towards architecture choices for digit recognition that can be folded into this architecture for greater improvement.

For images 3-5, the model was fairly adamant it had the correct classification (all were above 99%), and it was correct on each. These were 'clean' images, with good background/foreground contrast, and good lighting, there was no reason to believe an accurate model would mis-classify these images.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
* This item was not attempted


