package com.github.echo.network.structure.layer;

import com.github.echo.types.Activations;
import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;

import java.util.ArrayList;
import java.util.List;

public class DenseLayer {

    private final List<Neuron> neurons = new ArrayList<>();
    private final List<Synapse> synapses = new ArrayList<>();

    public DenseLayer(int neuronsCount, Activations activationFunction) {
        for (int i = 0; i < neuronsCount; i++) {
            neurons.add(new Neuron(activationFunction));
        }
    }

    /**
     * Creates synapses between neurons in this layer and neurons in the given next layer.
     *
     * @param  nextLayer  the layer to create synapses with, must not be null
     */
    public void createSynapses(DenseLayer nextLayer) {
        if (nextLayer == null) return;

        for (Neuron neuron : neurons) {
            for (Neuron nextNeuron : nextLayer.neurons()) {
                synapses.add(new Synapse(neuron, nextNeuron));
            }
        }
    }

    /**
     * Returns a list of Neuron objects representing the neurons in this layer.
     *
     * @return a list of Neuron objects
     */
    public List<Neuron> neurons() {
        return neurons;
    }

    /**
     * Returns the list of synapses in this layer.
     *
     * @return a list of synapses
     */
    public List<Synapse> synapses() {
        return synapses;
    }

    /**
     * Retrieves the neuron at the specified index in the list of neurons.
     *
     * @param index the index of the neuron to retrieve
     * @return the neuron at the specified index
     */
    public Neuron getNeuronAt(int index) {
        return neurons.get(index);
    }
}
