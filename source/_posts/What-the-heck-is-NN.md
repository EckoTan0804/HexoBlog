---
title: What the Heck is Neural Networks?
tags:
- What the heck
- Machine Learning	
- Neural Networks
mathjax: true

---

![Bildergebnis fÃ¼r Neural Network](https://cdn-images-1.medium.com/max/2000/1*bhFifratH9DjKqMBTeQG5A.gif)

<!--more-->

**Neural Networks (NNs)** are computing systems vaguely inspired by the [biological neural networks](https://en.wikipedia.org/wiki/Biological_neural_network) that constitute animal [brains](https://en.wikipedia.org/wiki/Brain). The neural network itself is not an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.

This article aims to give you a general knowledge of NNs, from bottom (neuron) to the top (neural networks).



## Artificial Neuron: the Basic Unit

As mentioned, neural networks were inspired by the neural architecture of a human brain, and like in a human brain the basic building block is called a **Neuron**. Its functionality is similar to a human neuron, i.e. it takes in some inputs and fires an output.

![Bildergebnis fÃ¼r Neural network Neuron](https://cdn-images-1.medium.com/max/1200/1*SJPacPhP4KDEB1AdhOFy_Q.png)

### How does a Artificial Neuron Work?

![Bildergebnis fÃ¼r neuron neural network](https://i.stack.imgur.com/VqOpE.jpg)

A neuron take its inputs, multiplies the inputs with their associated weights and sum them up. We call it the **pre-activation**. There is also another constant term called **bias** which will be added into the weighted sum. After obtaining the sum, the neuron applies an activation function to this and produce an activation (output). 

### Structure of a Artificial Neuron 

For a artificial neuron, the basic structure includes (Take a look at the image above):

+ inputs: $x_1, x_2, \dots, x_n$ 
+ with assoiciated weights/Parameters: $w_{1}, w_{2}, \dots, w_{n}$ 
+ bias: $b$
+ [activation function](https://en.wikipedia.org/wiki/Activation_function)
+ outputs: $y$

#### Weights/Parameters/Connections

Being the most important part of an NN, these (and the biases) are the numbers the NN has to learn in order to generalize to a problem. 

#### Bias

These numbers represent what the NN **“thinks”** it should add after multiplying the weights with the data. Of course, these are *always* wrong but the NN then learns the optimal biases as well.

> Sometimes the bias is set of default and is not drawn in the graph.

#### Activation Functions

##### Step Functions

A [step function](https://en.wikipedia.org/wiki/Step_function) is defined as

![img](https://cdn-images-1.medium.com/max/1600/1*0iOzeMS3s-3LTU9hYH9ryg.png) 

As one can see a step function is non-differentiable at zero. At present, a neural network uses back propagation method along with gradient descent to calculate weights of different layers. Since the step function is non-differentiable at zero hence it is not able to make progress with the [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) approach and fails in the task of updating the weights.

To overcome, this problem sigmoid functions were introduced instead of the step function.

##### **Sigmoid Function**

A [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) or logistic function is defined mathematically as

![img](https://cdn-images-1.medium.com/max/1600/1*MIeka59unAhS7MQk5e7FOg.png)

However, a sigmoid function also suffers from a problem of vanishing gradients. As can be seen from the picture a sigmoid function squashes it’s input into a very small output range [0,1] and has very steep gradients. Thus, there remain large regions of input space, where even a large change produces a very small change in the output. This is referred to as the problem of vanishing gradient. This problem increases with an increase in the number of layers and thus stagnates the learning of a neural network at a certain level.

##### **Tanh Function**

The [tanh](https://reference.wolfram.com/language/ref/Tanh.html) function is a rescaled version of the sigmoid, and its output range is $[ − 1,1]$ instead of $[0,1]$.

![img](https://cdn-images-1.medium.com/max/1600/1*Ouueb-J2gBRvA-M_1rsXZA.png)

The general reason for using a Tanh function in some places instead of the sigmoid function is because since data is centered around 0, the derivatives are higher. A higher gradient helps in a better learning rate. Below attached are plotted gradients of two functions tanh and sigmoid.

However, the problem of vanishing gradients still persists in Tanh function.

##### **ReLU Function**

The [Rectified Linear Unit](https://reference.wolfram.com/language/ref/Tanh.html) is the most commonly used activation function in deep learning models. The function returns 0 if it receives any negative input, but for any positive value x, it returns that value back. So, it can be written as f(x)=max (0, x).

![img](https://cdn-images-1.medium.com/max/1600/1*njuH4XVXf-l9pR_RorUOrA.png)

The **Leaky ReLU** is one of the most well-known. It is the same as ReLU for positive numbers. But instead of being 0 for all negative values, it has a constant slope (less than 1).

> That slope is a parameter the user sets when building the model, and it is frequently called $\alpha$. For example, if the user sets $α=0.3$, the activation function is
> $$
> f(x) = max (0.3 \cdot x,  x)
> $$
>
>
> This has the theoretical advantage that, by being influenced, by `x` at all values, it may make more complete use of the information contained in x.

There are other alternatives, but both practitioners and researchers have generally found an insufficient benefit to justify using anything other than ReLU. In general practice as well, ReLU has found to be performing better than sigmoid or tanh functions.

### Closer Look at the Neuron

Let's get a little bit deeper and see how a neuron works mathematically. 

![neuron-diagram-01](https://themenwhostareatcodes.files.wordpress.com/2014/03/neuron-diagram-01.png?w=450&h=264)

All this given, we can describe the neuron $k$ by the pair of equations:
$$
v_k = \sum_{j=1}^{m}{w_{kj}x_j}
$$

$$
y_k = \varphi(v_k + b_k)
$$

Sometimes we consider the bias $b_k$ as the fixed input $x_0$ with weight $w_{k0} = 1$

![neuron-diagram-02](https://themenwhostareatcodes.files.wordpress.com/2014/03/neuron-diagram-02.png?w=450&h=273)

We can also describe the neuron $k$ using the matrices:

+  Input matrix: $ \vec{x} = (x_1, x_2, \dots, x_m) ^ {\mathsf{T}}$
+ Weight matrix: $W = (w_{k1}, w_{k2}, \dots, w_{km})$

In this case, 
$$
y_k = \varphi(W \cdot \vec{x} + b_k)
$$
From linear algebra's point of view, the behaviour of a neuron consists of:

1. A [linear transformation](https://en.wikipedia.org/wiki/Linear_map) by the “weight” matrix $W$
2. A [translation](https://en.wikipedia.org/wiki/Translation_(geometry)) by the vector $b_k$
3. Point-wise application of activation function $\varphi$.

For me I prefer this way of expression. The reasons are:

+ It is much more succinct when the NN is big and complicated
+ Matrix is exactly the representation of linear mapping/transformation. And linear transformation is also what a neuron does.



## Perceptron: Single-Layer NN

![img](https://cdn-images-1.medium.com/max/1600/1*-JtN9TWuoZMz7z9QKbT85A.png)

First of all, we have to be clear that the layer we mention refers to the layer between the input nodes and the output nodes. Which means, we are counting the number of the layers of links/edges/weight matrices.

Therefore, [perceptron](http://en.wikipedia.org/wiki/Perceptron) is often called a **single-layer** network on account of having only **1** layer of links (**1** weight matrix), between input and output.

![img](https://cdn-images-1.medium.com/max/1600/1*n6sJ4yZQzwKL9wnF5wnVNg.png)

In the modern sense, the perceptron is an algorithm for learning a binary classifier: a function that maps its input $x$ (a real-valued vector) to an output value $f(x)\in \{0, 1\} $ (a single binary value):
$$
\begin{equation}
    f(x)=
    \begin{cases}
        1& \text{if } w\cdot x + b > 0\\\\
        0& \text{else}
    \end{cases}
\end{equation}
$$
The value of $f(x)$ (0 or 1) is used to classify **x** as either a positive or a negative instance, in the case of a binary classification problem. 



### Boolean Functions Using Perceptron

Let's take a look at the fundamental boolean functions. Which boolean functions can be solved by the perceptron?

#### OR Function — Can Do!✅

![img](https://cdn-images-1.medium.com/max/1600/1*C5LeL8JDfoGbkUg0cu1M-w.png)

The above "possible solution" was obtained by solving the linear system of equations on the left. It is clear that the solution separates the input space into two spaces, negative and positive half spaces.

You can try it out by yourself for [**AND**](https://en.wikipedia.org/wiki/Logical_conjunction) and [**NOT**](https://en.wikipedia.org/wiki/Negation) function. They both can be solved by perceptron.

You may ask, is there any boolean functions that the perceptron can not solve?

Unfortunately, there is.

#### XOR Function — Can’t Do!❌

Now let's look at a non-linear boolean function, i.e. [**XOR**](https://en.wikipedia.org/wiki/Exclusive_or), you can not draw a line to separate positive inputs from the negative ones.

![img](https://cdn-images-1.medium.com/max/1600/1*E7YhKsJ2wb-VdeXpg89lvg.png)

Notice that the fourth equation contradicts the second and the third equation. Point is, there are no *perceptron* solutions for non-linearly separated data.

Therefore, the main problem of a **single** perceptron is that <span style="color:red">**a single *perceptron* cannot learn to separate the data that are non-linear in nature**</span>.



## Multilayer Perceptron (MLP)

![img](https://cdn-images-1.medium.com/max/1600/1*w4wR9mLS46s3gsX_7pW_ww.jpeg)

As a single layer perceptron can not separate data that are non-linear, can we combine multiple perceptrons to make it possible?

YES! Here comes the **Multilayer perceptron (MLP)**!

The multilayer perceptron, the hello world of deep learning, composes of more than one perceptron, is a deep artificial neural network. It consists of 

+ an input layer to receive the signal, 
+ an output layer that makes a decision or prediction about the input,
+ in between those two, an **arbitrary number** of hidden layers that are the true computational engine of the MLP. 

![img](https://1.bp.blogspot.com/-Xal8aZ5MDL8/WlJm8dh1J9I/AAAAAAAAAo4/uCj6tt4T3T0HHUY4uexNuq2BXTUwcChqACLcBGAs/s400/Multilayer-Perceptron.jpg)



A MLP is a class of [feedforward](https://en.wikipedia.org/wiki/Feedforward_neural_network) artificial neural network. Feedforward means between the nodes do not form a cycle. As you can see in the image before, the data flow from the input layer to hidden layers, then finally to the output layer. 

Multilayer perceptrons are often applied to [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) problems: they train on a set of input-output pairs and learn to model the correlation (or dependencies) between those inputs and outputs. Training involves adjusting the parameters, or the weights and biases, of the model in order to minimize error. [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) is used to make those weigh and bias adjustments relative to the error, and the error itself can be measured in a variety of ways, including by root mean squared error (RMSE). We will talk about it later.

### Example: Power of the MLP

MLPs with one hidden layer are capable of approximating any continuous function.

Let's look at an example from [colah's blog](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/). 

Start with a very simple dataset, two curves on a plane. The network will learn to classify points as belonging to one or the other.

![img](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/simple2_data.png)

The obvious way to visualize the behavior of a neural network – or any classification algorithm, for that matter – is to simply look at how it classifies every possible data point.

We’ll start with the simplest possible class of neural network, one with only an input layer and an output layer (also the single layer perceptron). Such a network simply tries to separate the two classes of data by dividing them with a line.

![img](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/simple2_linear.png)

As you can see, its performance is not so good as we expect. So let's just add one layer between the input and the output layer. Now it separates the data with a more complicated curve than a line.

![img](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/simple2_0.png)

With each layer, the network transforms the data, creating a new *representation*. We can look at the data in each of these representations and how the network classifies them. When we get to the final representation, the network will just draw a line through the data (or, in higher dimensions, a hyperplane).

In the previous visualization, we looked at the data in its “raw” representation. You can think of that as us looking at the input layer. Now we will look at it after it is transformed by the first layer. You can think of this as us looking at the hidden layer.

Each dimension corresponds to the firing of a neuron in the layer.

![img](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/simple2_1.png)



## Last but not Least

I hope you have already gained some general knowledge of NN after reading this article. You may have a number of questions: What is backpropagation? how it works? How many types of neural networks are there? What is gradient descent? ... Don't worry, we will talk about it detailedly in the later post.

**Happy reading, happy learning!**



## Reference

+ [Neural Networks: All YOU Need to Know](https://towardsdatascience.com/nns-aynk-c34efe37f15a)
+ [Understanding Neural Networks. From neuron to RNN, CNN, and Deep Learning](https://towardsdatascience.com/understanding-neural-networks-from-neuron-to-rnn-cnn-and-deep-learning-cd88e90e0a90)

+ [Perceptron: The Artificial Neuron (An Essential Upgrade To The McCulloch-Pitts Neuron)](https://towardsdatascience.com/perceptron-the-artificial-neuron-4d8c70d5cc8d)
+ [Neural Networks, Manifolds, and Topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

+ [Gradient Descent: All You Need to Know](https://hackernoon.com/gradient-descent-aynk-7cbe95a778da)
+ [A Beginner's Guide to Multilayer Perceptrons (MLP)](https://skymind.ai/wiki/multilayer-perceptron)