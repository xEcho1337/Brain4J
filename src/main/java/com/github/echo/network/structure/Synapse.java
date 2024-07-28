package com.github.echo.network.structure;

public class Synapse {

    private final Neuron inputNeuron;
    private final Neuron outputNeuron;
    private double weight = 2 * Math.random() - 1;

    public Synapse(Neuron inputNeuron, Neuron outputNeuron) {
        this.inputNeuron = inputNeuron;
        this.outputNeuron = outputNeuron;
    }

    public Neuron inputNeuron() {
        return inputNeuron;
    }

    public Neuron outputNeuron() {
        return outputNeuron;
    }

    public double weight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}
