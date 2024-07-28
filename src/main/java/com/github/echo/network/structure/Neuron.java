package com.github.echo.network.structure;

import com.github.echo.activations.Activations;

public class Neuron {

    private final Activations activationFunction;
    private double bias = (2 * Math.random()) - 1;
    private double value;

    public Neuron(Activations activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double bias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double value() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public void applyFunction() {
        this.value = activationFunction.activationFunction().apply(value - bias);
    }
}
