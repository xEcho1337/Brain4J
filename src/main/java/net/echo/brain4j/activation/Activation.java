package net.echo.brain4j.activation;

import net.echo.brain4j.structure.Neuron;

import java.util.List;

public interface Activation {

    double activate(double input);

    double[] activateMultiple(double[] input);

    double getDerivative(double input);

    void apply(List<Neuron> neurons);
}
