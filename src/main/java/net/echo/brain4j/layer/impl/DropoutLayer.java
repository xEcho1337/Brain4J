package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;

import java.util.List;

public class DropoutLayer extends Layer {

    private final double dropout;

    public DropoutLayer(double dropout) {
        super(0, Activations.LINEAR);

        if (dropout <= 0 || dropout > 1) {
            throw new IllegalArgumentException("Dropout must be between 0 and 1");
        }

        this.dropout = dropout;
    }

    public void process(List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            if (Math.random() < dropout) {
                neuron.setValue(0);
            }
        }
    }
    public void backward(List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            neuron.setValue(neuron.getValue() * (1.0 - dropout));
        }
    }

    public double getDropout() {
        return dropout;
    }
}
