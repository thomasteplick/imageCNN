<h3>Convolutional Neural Network (CNN) using a Multilayer Perceptron (MLP) with the Back-Propagation Algorithm for Image Recognition</h3>
<hr>
This program is a web application written in Go that makes extensive use of the html/template package.
Navigate to the C:\Users\your-name\ImageCNN\src\convolution\ directory and issue "go run imagecnn.go" to
start the Convolutional Neural Network server. In a web browser enter http://127.0.0.1:8080/imagCNN
in the address bar.  There are two phases of operation:  the training phase and the testing phase.  During the training
phase, examples consisting of alpha-numeric characters and the desired class are supplied to the network.
The network itself is a directed graph consisting of an input layer of nodes, one or more hidden layers of nodes, and
an output layer of nodes.  Each layer of nodes can be arbitrarily deep.  The nodes of the network are connected by weighted
links.  The network is fully connected.  This means that every node is connected to its immediately adjacent neighbor node.  The weights are trained
by first propagating the inputs forward, layer by layer, to the output layer of nodes.  The output layer of nodes finds the
difference between the desired and its output and back propagates the errors to the input layer.  The hidden and input layer
weights are assigned “credit” for the errors by using the chain rule of differential calculus.  Each neuron consists of a
linear combiner and an activation function.  This program uses the hyperbolic tangent function to serve as the activation function.
This function is non-linear and differentiable and limits its output to be between -1 and 1.  <b>The purpose of this program is to classify alpha-numeric
characters in an image</b>.  The image contains 1089 randomly-placed characters.  The CNN is first trained using images of the 64 alpha-numeric characters
plus "+" and "/".  The Learning Curve of mean-square error (MSE) versus epoch shows the results of this phase.  In the testing phase, the weights obtained
from the training phase are used to classify the 1089 characters that are randomly placed in an image.  The image is scanned with forward propagation and
each sub-image is classified.  The results of the classification are tabulated and the image is displayed.  In addition, a PNG image is constructed from 
the image and saved  to the data/ directory.
<br/>
<p>
The user selects the CNN-MLP training parameters:
<li>Hidden Layers</li>
<li>Layer Depth</li>
<li>Learning Rate</li>
<li>Momentum</li>
<li>Epochs</li>
</p>
<p>
The <i>Learning Rate</i> and <i>Momentum</i> must be less than one.  Each <i>Epoch</i> consists of the number of <i>Training Examples</i>.  
One training example is an alpha-numeric character and the desired class (0, 1,…, 63).  There are 64 characters and therefore 64 classes.
The characters are a sequence of eighty-one 1 and -1 integers that represent the encoded image of the character.  The file containing the
encoded characters was produced by the character encoding repository program at thomasteplick/char-encoder.
</p>
<p>
When the <i>Submit</i> button on the CNN Training Parameters form is clicked, the weights in the network are trained
and the Learning Curve consisting of mean-square error (MSE) versus Epoch is displayed.  As can be seen in the screen shots below, 
there is significant variance over the ensemble, but it eventually settles down after about 20 epochs. An epoch is the forward
and backward propagation of all the 64 training samples.
</p>
<p>
When the <i>Test</i> link is clicked, 64 examples are supplied to the CNN-MLP.  It classifies the alpha-numeric characters.
The test results are tabulated and the actual characters are graphed from the encoding that was supplied to the CNN-MLP.
It takes some trial-and-error with the CNN Training Parameters to reduce the MSE to zero.  It is possible to a specify a 
more complex MLP than necessary and not get good results.  For example, using more hidden layers, a greater layer depth,
or over training with more examples than necessary may be detrimental to the CNN-MLP.  Clicking the <i>Train</i> link starts a new training
phase and the MLP Training Parameters must be entered again.
</p>

<b>CNN-MLP Learning Curve, MSE vs Epoch, 1 Hidden Layer, Hidden Layer Depth = 15</b>

![image](https://github.com/thomasteplick/imageCNN/assets/117768679/98b790da-4cec-4b66-93f2-314697ef27e5)

<b>CNN-MLP Image Recognition Test Results, 1 Hidden Layer, Hidden Layer Depth = 15</b>

![image](https://github.com/thomasteplick/imageCNN/assets/117768679/a0da68a2-9212-470c-8cd2-83eeb399768c)
![image](https://github.com/thomasteplick/imageCNN/assets/117768679/acbae280-ae3d-4730-b79d-8892e1d9e056)

<b>image.png, 300x300</b>

![image](https://github.com/thomasteplick/imageCNN/assets/117768679/b11d0145-587f-48e6-a759-26a215dc217c)

