package com.github.echo.activations;

import java.util.function.Function;

public enum Activations {

    RELU(x -> Math.max(x, 0)),
    SIGMOID(x -> 1 / (1 + Math.exp(-x))),
    TANH(Math::tanh),
    LINEAR(x -> x);

    private final Function<Double, Double> activationFunction;

    /**
     * Defines the activation function that can be used in the neural network.
     *
     * @param activationFunction the activation function
     */
    Activations(Function<Double, Double> activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Returns the function associated to the activation function.
     *
     * @return a function that takes a double as input and returns a double as output
     */
    public Function<Double, Double> activationFunction() {
        return activationFunction;
    }
}
