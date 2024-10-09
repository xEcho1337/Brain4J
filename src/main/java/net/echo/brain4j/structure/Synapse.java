package net.echo.brain4j.structure;

public class Synapse {

    private final Neuron inputNeuron;
    private final Neuron outputNeuron;
    private double weight;

    public Synapse(Neuron inputNeuron, Neuron outputNeuron, double bound) {
        this.inputNeuron = inputNeuron;
        this.outputNeuron = outputNeuron;
        this.weight = (Math.random() * 2 * bound) - bound;
    }

    public Neuron getInputNeuron() {
        return inputNeuron;
    }

    public Neuron getOutputNeuron() {
        return outputNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }
}
