package com.github.echo.network.structure.layer;

import com.github.echo.activations.Activations;
import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;

import java.util.ArrayList;
import java.util.List;

public class Layer {

    private final List<Neuron> neurons = new ArrayList<>();
    private final List<Synapse> synapses = new ArrayList<>();

    public Layer(int neuronsCount, Activations activationFunction) {
        for (int i = 0; i < neuronsCount; i++) {
            neurons.add(new Neuron(activationFunction));
        }
    }

    public void createSynapses(Layer nextLayer) {
        if (nextLayer == null) return;

        for (Neuron neuron : neurons) {
            for (Neuron nextNeuron : nextLayer.neurons()) {
                synapses.add(new Synapse(neuron, nextNeuron));
            }
        }
    }

    public List<Neuron> neurons() {
        return neurons;
    }

    public List<Synapse> synapses() {
        return synapses;
    }

    public Neuron getNeuronAt(int index) {
        return neurons.get(index);
    }
}
