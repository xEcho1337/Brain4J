package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;

import java.util.List;

public class SigmoidActivation implements Activation {

    @Override
    public double activate(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double[] activateMultiple(double[] input) {
        double[] result = new double[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = activate(input[i]);
        }
        return result;
    }

    @Override
    public double getDerivative(double input) {
        return activate(input) * (1 - activate(input));
    }

    @Override
    public void apply(List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double output = activate(neuron.getValue() + neuron.getBias());

            neuron.setValue(output);
        }
    }
}
