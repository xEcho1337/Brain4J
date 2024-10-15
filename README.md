# Brain4J

[![Brain4J](https://img.shields.io/badge/Brain4J-2.0.11-blue.svg)](https://github.com/brain4j/brain4j)

Brain4J is Java library for Machine Learning, designed to be as light as possible and easily portable.


## Getting started

When making a neural network there are plenty of options you can choose from. This time we are making a simple NN to 
calculate the XOR gate.

Firstly, we need to define the model, for a XOR gate we can use a really simple one, made by only 4 layers.

```java
Model network = new FeedForwardModel(
        new DenseLayer(2, Activations.LINEAR),
        new DenseLayer(4, Activations.RELU),
        new DenseLayer(4, Activations.RELU),
        new DenseLayer(1, Activations.SIGMOID)
);
```

At this point, we need to define the weight initialization method and the loss function that will be used when training. 
We can achieve this by calling the compile method like below:

```java
network.compile(InitializationType.XAVIER, LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.001));
```

For networks with only one output neuron (where the output is between 0 and 1) we use Binary Cross Entropy,
with Adaptive Moment Estimation (Adam).

Now we need to define our training data by using `DataSet` and `DataRow`.

```java
DataRow first = new DataRow(new double[]{0.0, 0.0}, 0.0);
DataRow second = new DataRow(new double[]{0.0, 1.0}, 1.0);
DataRow third = new DataRow(new double[]{1.0, 0.0}, 1.0);
DataRow fourth = new DataRow(new double[]{1.0, 1.0}, 0.0);

DataSet training = new DataSet(first, second, third, fourth);
```

We have everything setup, we can call the fit method inside a loop and wait for the network to finish.

```java
double error;
        
do {
    error = network.fit(training);
} while (error > 0.01);
```

The code above will train the neural network with a learning rate of 0.01, and it will stop only when it achieves an 
error of less than 1%.

## Contributing & Contact

Pull requests and issues are always welcome. Do not hesitate to contact [@nettyfan](https://t.me/nettyfan) on telegram if you need further explanation
