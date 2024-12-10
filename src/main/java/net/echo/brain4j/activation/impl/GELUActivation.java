package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;

import java.util.List;

public class GELUActivation implements Activation {

    @Override
    public double activate(double input) {
        return 0.5 * input * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715 * Math.pow(input, 3))));
    }

    @Override
    public double[] activateMultiple(double[] input) {
        throw new UnsupportedOperationException("GELU activation function is not supported for multiple inputs.");
    }

    @Override
    public double getDerivative(double input) {
        double tanhTerm = Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715 * Math.pow(input, 3)));
        return 0.5 * (1 + tanhTerm) + 0.5 * input * (1 - Math.pow(tanhTerm, 2)) * Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * Math.pow(input, 2));
    }

    @Override
    public void apply(List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            neuron.setValue(activate(neuron.getValue()));
        }
    }
}
