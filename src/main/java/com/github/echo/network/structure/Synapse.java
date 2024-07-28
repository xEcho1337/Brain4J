package com.github.echo.network.structure;

public class Synapse {

    private final Neuron inputNeuron;
    private final Neuron outputNeuron;
    private double weight = 2 * Math.random() - 1;

    public Synapse(Neuron inputNeuron, Neuron outputNeuron) {
        this.inputNeuron = inputNeuron;
        this.outputNeuron = outputNeuron;
    }

    /**
     * Returns the input neuron of the synapse.
     *
     * @return the input neuron of the synapse
     */
    public Neuron inputNeuron() {
        return inputNeuron;
    }

    /**
     * Returns the output neuron of this synapse.
     *
     * @return the output neuron of this synapse
     */
    public Neuron outputNeuron() {
        return outputNeuron;
    }

    /**
     * Returns the weight of the synapse.
     *
     * @return the weight of the synapse
     */
    public double weight() {
        return weight;
    }

    /**
     * Sets the weight of the synapse.
     *
     * @param weight the new weight value to be set
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }
}
