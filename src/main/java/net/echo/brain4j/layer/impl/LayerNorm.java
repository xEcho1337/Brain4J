package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public class LayerNorm extends Layer {

    private final double epsilon;

    public LayerNorm() {
        this(1e-5);
    }

    public LayerNorm(double epsilon) {
        super(0, Activations.LINEAR);
        this.epsilon = epsilon;
    }

    @Override
    public void applyFunction(Layer previous) {
        List<Neuron> inputs = previous.getNeurons();

        double mean = calculateMean(inputs);
        double variance = calculateVariance(inputs, mean);

        for (Neuron input : inputs) {
            double value = input.getLocalValue();
            double normalized = (value - mean) / Math.sqrt(variance + epsilon);

            input.setValue(normalized);
        }
    }

    public Vector normalize(Vector input) {
        double mean = input.mean();
        double variance = input.variance(mean);

        double denominator = Math.sqrt(variance + epsilon);

        for (int i = 0; i < input.size(); i++) {
            double value = input.get(i);
            double normalized = (value - mean) / denominator;

            input.set(i, normalized);
        }

        return input;
    }

    private double calculateMean(List<Neuron> inputs) {
        double sum = 0.0;

        for (Neuron value : inputs) {
            sum += value.getLocalValue();
        }

        return sum / inputs.size();
    }

    private double calculateVariance(List<Neuron> inputs, double mean) {
        double sum = 0.0;

        for (Neuron value : inputs) {
            sum += Math.pow(value.getLocalValue() - mean, 2);
        }

        return sum / inputs.size();
    }
}
