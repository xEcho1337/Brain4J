package com.github.echo.network;

import com.github.echo.network.structure.layer.Layer;
import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final List<Layer> layers = new ArrayList<>();

    /**
     * Adds a layer to the neural network and returns the updated NeuralNetwork object.
     *
     * @param  layer  the layer to be added to the neural network
     * @return        the updated NeuralNetwork object with the added layer
     */
    public NeuralNetwork addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Builds the neural network by creating synapses between each pair of adjacent layers.
     *
     * @return the built neural network
     */
    public NeuralNetwork build() {
        resetSynapses();

        return this;
    }

    /**
     * Resets the synapses in the neural network and creates new ones between each pair of adjacent layers.
     */
    public void resetSynapses() {
        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);

            // Reset all the synapses
            layer.synapses().clear();

            if (l == layers.size() - 1) break;

            Layer nextLayer = layers.get(l + 1);

            layer.createSynapses(nextLayer);
        }
    }

    /**
     * Calculates the output of the neural network given an input array.
     *
     * @param input the input array to the neural network
     * @return an array of doubles representing the output of the neural network
     * @throws IllegalArgumentException if no layers are defined or if the input layer size does not match the input size
     */
    public double[] calculateOutput(double[] input) {
        if (layers.isEmpty()) {
            throw new IllegalArgumentException("No layers defined");
        }

        Layer inputLayer = layers.get(0);

        if (inputLayer.neurons().size() != input.length) {
            throw new IllegalArgumentException("Input layer size does not match input size");
        }

        // Reset every neuron value
        for (Layer layer : layers) {
            for (Neuron neuron : layer.neurons()) {
                neuron.setValue(0);
            }
        }

        // Inserts the starting value of the input layer
        for (int i = 0; i < input.length; i++) {
            inputLayer.getNeuronAt(i).setValue(input[i]);
        }

        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);

            // We reached the output layer, which already has everything calculated
            if (l + 1 == layers.size()) break;

            Layer nextLayer = layers.get(l + 1);

            for (Synapse synapse : layer.synapses()) {
                Neuron inputNeuron = synapse.inputNeuron();
                Neuron outputNeuron = synapse.outputNeuron();

                // Weighted sum
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

    /**
     * Returns the list of layers in the neural network.
     *
     * @return a list of Layer objects representing the layers in the neural network
     */
    public List<Layer> layers() {
        return layers;
    }

    /**
     * Retrieves the Layer object at the specified index from the layers list.
     *
     * @param  index  the index of the Layer object to retrieve
     * @return        the Layer object at the specified index
     * @throws IndexOutOfBoundsException if the index is out of range
     *                                   (index < 0 || index >= size())
     */
    public Layer getLayerAt(int index) {
        return layers.get(index);
    }
}
