package com.github.echo.network.structure;

import com.github.echo.types.Activations;

public class Neuron {

    private final Activations activationFunction;
    private double bias = 2 * Math.random() - 1;
    private double value;
    private double delta;

    public Neuron(Activations activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Returns the activation function associated with this neuron.
     *
     * @return the activation function for this neuron
     */
    public Activations activationFunction() {
        return activationFunction;
    }

    /**
     * Returns the bias value of the neuron.
     *
     * @return the bias value of the neuron
     */
    public double bias() {
        return bias;
    }

    /**
     * Sets the bias value of the neuron.
     *
     * @param  bias  the new bias value
     */
    public void setBias(double bias) {
        this.bias = bias;
    }

    /**
     * Returns the value of the neuron.
     *
     * @return the value of the neuron
     */
    public double value() {
        return value;
    }

    /**
     * Sets the value of the neuron.
     *
     * @param  value  the new value of the neuron
     */
    public void setValue(double value) {
        this.value = value;
    }

    /**
     * Returns the delta value of the Neuron.
     *
     * @return the delta value
     */
    public double delta() {
        return delta;
    }

    /**
     * Sets the delta value of the neuron.
     *
     * @param  delta  the new delta value
     */
    public void setDelta(double delta) {
        this.delta = delta;
    }

    /**
     * Applies the activation function to the value of the neuron, subtracting the bias, and assigns the result to the value field.
     */
    public void applyFunction() {
        this.value = activationFunction.function().apply(value + bias);
    }
}
