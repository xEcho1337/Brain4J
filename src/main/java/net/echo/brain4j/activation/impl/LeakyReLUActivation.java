package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;

import java.util.List;

public class LeakyReLUActivation implements Activation {

    @Override
    public double activate(double input) {
        return Math.max(0.01 * input, input);
    }

    @Override
    public double[] activateMultiple(double[] input) {
        throw new UnsupportedOperationException("Leaky ReLU activation function is not supported for multiple inputs");
    }

    @Override
    public double getDerivative(double input) {
        return input > 0 ? 1 : 0.01;
    }

    @Override
    public void apply(List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double output = activate(neuron.getValue() + neuron.getBias());

            neuron.setValue(output);
        }
    }
}
