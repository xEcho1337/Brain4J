package com.github.echo.network;

import com.github.echo.network.structure.layer.Layer;
import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final List<Layer> layers = new ArrayList<>();

    public NeuralNetwork addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public NeuralNetwork build() {
        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);

            if (l == layers.size() - 1) break;

            Layer nextLayer = layers.get(l + 1);

            layer.createSynapses(nextLayer);
        }

        return this;
    }

    public double[] calculateOutput(double[] input) {
        if (layers.isEmpty()) {
            throw new IllegalArgumentException("No layers defined");
        }

        Layer inputLayer = layers.get(0);

        if (inputLayer.neurons().size() != input.length) {
            throw new IllegalArgumentException("Input layer size does not match input size");
        }

        for (int i = 0; i < input.length; i++) {
            inputLayer.getNeuronAt(i).setValue(input[i]);
        }

        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);

            if (l + 1 == layers.size()) break;

            Layer nextLayer = layers.get(l + 1);

            for (Synapse synapse : layer.synapses()) {
                Neuron inputNeuron = synapse.inputNeuron();
                Neuron outputNeuron = synapse.outputNeuron();

                outputNeuron.setValue(outputNeuron.value() + inputNeuron.value() * synapse.weight());
            }

            // Apply the activation function
            for (Neuron neuron : nextLayer.neurons()) {
                neuron.applyFunction();
            }
        }

        Layer outputLayer = layers.get(layers.size() - 1);
        double[] output = new double[outputLayer.neurons().size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeuronAt(i).value();
        }

        return output;
    }

    public List<Layer> layers() {
        return layers;
    }

    public Layer getLayerAt(int index) {
        return layers.get(index);
    }
}
