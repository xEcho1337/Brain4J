package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;

import java.util.List;

public class TanhActivation implements Activation {

    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double[] activateMultiple(double[] inputs) {
        double[] result = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            result[i] = activate(inputs[i]);
        }
        return result;
    }

    @Override
    public double getDerivative(double input) {
        return 1.0 - Math.pow(Math.tanh(input), 2);
    }

    @Override
    public void apply(List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double value = neuron.getValue() + neuron.getBias();
            neuron.setValue(activate(value));
        }
    }
}
