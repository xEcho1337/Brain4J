package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;

import java.util.List;

public class ELUActivation implements Activation {

    private final double alpha = 1.0;

    @Override
    public double activate(double input) {
        if (input > 0) {
            return input;
        } else {
            return alpha * (Math.exp(input) - 1);
        }
    }

    @Override
    public double[] activateMultiple(double[] input) {
        throw new UnsupportedOperationException("ELU activation function is not supported for multiple inputs.");
    }

    @Override
    public double getDerivative(double input) {
        if (input > 0) {
            return 1;
        } else {
            return alpha * Math.exp(input);
        }
    }

    @Override
    public void apply(List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            neuron.setValue(activate(neuron.getValue()));
        }
    }
}
