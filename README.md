## Dog-breed-classifier using CNN and transfer learning in PyTorch
This is the README.md for Dog breed recogntion project. 

### Project Overview
<p align="justify">
The goal of the project is to build a machine learning model that can be used within
web/phone app to process real-world, user-supplied images. The algorithm has to perform
two tasks:</p>
<ul>
<li>Given an image of a dog, the algorithm will identify and recognize
it's breed.</li>
<li>If the supplied image is of a human, the code will identify the
resembling dog breed.</li>
</ul>
<p align="justify">For performing this multiclass classification, we can use <b>Convolutional Neural Network</b> to solve the problem.The solution involves three steps. First, to detect
human images, we can use existing algorithm like OpenCV’s implementation of
Haar feature based cascade classifiers. Second, to detect dog-images we will use a
pretrained VGG16 model. Finally, after the image is identified as dog/human, we
can pass this image to an CNN model which will process the image and predict the
breed that matches the best out of 133 breeds. </p>

### CNN model created from scratch
<p align="justify">The CNN model from scratch was built to solve the problem. The model has 5
convolutional layers. All convolutional layers have kernel size of 3 and stride 1 except the first layer with kernel 11 and stride 4. The first conv layer (conv1) takes the input 3 channel image depth that gives at the final conv layer (conv5) an output depth of 256. Batch normalization layer and ReLU activation function is used for every convolution layer. Max pooling layer of kernel = 3 and stride = 2 is used for the first and second convolution layers separately while the 3rd, 4th and 5th conv layers block has one max pooling layer. In addition, an average pooling layer separates the convolution layer and the fully connected layer. Moreover, we have three fully connected layers that finally produces 133-dimensional output. A dropout of 0.25 is added for avoiding over overfitting.</p>

### Refinement - CNN model created with transfer learning
<p align="justify">The CNN created from scratch have accuracy of 24%, Though it meets the benchmarking, the model can be significantly improved by using transfer learning. To create <b>CNN with transfer learning</b>, I have selected the <b>Resnet50 architecture</b>
which is pre-trained on ImageNet dataset, the architecture is 50 layers deep. The
last convolutional output of Resnet50 is fed as input to our model. We only need
to add a fully connected layer to produce 133-dimensional output (one for each
dog category). The model performed extremely well when compared to CNN from
scratch. With just 10 epochs, the model got 85% accuracy.</p>

![Sample output](./sample_output.PNG) 

### Model Evaluation
<p align="justify">The CNN model created using transfer learning with
ResNet50 architecture was trained for 10 epochs, and the final model produced an
accuracy of 85% on test data. The model correctly predicted breeds for 716 images out of 836 total images.</p>

**Accuracy on test data: 85% (716/836)**


