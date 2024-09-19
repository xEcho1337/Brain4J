package com.github.echo.network;

import com.github.echo.network.structure.layer.DenseLayer;
import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;
import com.github.echo.network.structure.layer.OutputLayer;
import com.github.echo.types.lost.LossFunction;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final List<DenseLayer> layers = new ArrayList<>();

    public NeuralNetwork() {
    }

    public NeuralNetwork(NeuralNetwork parent) {
        for (DenseLayer layer : parent.layers()) {
            addLayer(layer);
        }
    }

    /**
     * Adds a layer to the neural network and returns the updated NeuralNetwork object.
     *
     * @param  layer  the layer to be added to the neural network
     * @return        the updated NeuralNetwork object with the added layer
     */
    public NeuralNetwork addLayer(DenseLayer layer) {
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
        if (layers.size() < 2) {
            throw new IllegalStateException("The neural network doesn't have enough layers! THe minimum is 2.");
        }

        DenseLayer lastLayer = layers.get(layers.size() - 1);

        if (!(lastLayer instanceof OutputLayer)) {
            throw new IllegalArgumentException("The last layer should be an instance of OutputLayer!");
        }

        for (int l = 0; l < layers.size(); l++) {
            DenseLayer layer = layers.get(l);

            // Reset all the synapses
            layer.synapses().clear();

            if (l == layers.size() - 1) break;

            DenseLayer nextLayer = layers.get(l + 1);

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

        DenseLayer inputLayer = layers.get(0);

        if (inputLayer.neurons().size() != input.length) {
            throw new IllegalArgumentException("Input layer size does not match input size");
        }

        // Reset every neuron value
        for (DenseLayer layer : layers) {
            for (Neuron neuron : layer.neurons()) {
                neuron.setValue(0);
            }
        }

        // Inserts the starting value of the input layer
        for (int i = 0; i < input.length; i++) {
            inputLayer.getNeuronAt(i).setValue(input[i]);
        }

        for (int l = 0; l < layers.size(); l++) {
            DenseLayer layer = layers.get(l);

            // We reached the output layer, which already has everything calculated
            if (l + 1 == layers.size()) break;

            DenseLayer nextLayer = layers.get(l + 1);

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

        DenseLayer outputLayer = layers.get(layers.size() - 1);
        double[] output = new double[outputLayer.neurons().size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeuronAt(i).value();
        }

        return output;
    }

    /**
     * Calculates the output of the neural network at the given index layer, given an input array.
     *
     * @param  inputs the input array to the neural network
     * @param  index  the index of the layer to calculate the output for
     * @return        an array of doubles representing the output of the neural network at the given index layer
     * @throws IllegalArgumentException if the input layer size does not match the input size
     * @throws IndexOutOfBoundsException if the index is out of range (index < 0 || index >= size())
     */
    public double[] forward(double[] inputs, int index) {
        NeuralNetwork clone = new NeuralNetwork(this);

        DenseLayer layer = clone.layers().get(index);
        DenseLayer previous = clone.layers().get(index - 1);

        if (previous.neurons().size() != inputs.length) {
            throw new IllegalArgumentException("Input layer size does not match input size");
        }

        double[] outputs = new double[layer.neurons().size()];

        for (int i = 0; i < previous.neurons().size(); i++) {
            Neuron neuron = previous.getNeuronAt(i);
            neuron.setValue(inputs[i]);
        }

        for (int i = 0; i < layer.neurons().size(); i++) {
            Neuron outputNeuron = layer.getNeuronAt(i);

            for (Synapse synapse : layer.synapses()) {
                Neuron inputNeuron = synapse.inputNeuron();
                Neuron neuron = synapse.outputNeuron();

                if (neuron.equals(outputNeuron)) {
                    outputNeuron.setValue(outputNeuron.value() + inputNeuron.value() * synapse.weight());
                }
            }

            outputNeuron.applyFunction();
            outputs[i] = outputNeuron.value();
        }

        return outputs;
    }

    /**
     * Returns the list of layers in the neural network.
     *
     * @return a list of Layer objects representing the layers in the neural network
     */
    public List<DenseLayer> layers() {
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
    public DenseLayer getLayerAt(int index) {
        return layers.get(index);
    }


    public OutputLayer getOutputLayer() {
        DenseLayer layer = layers.get(layers.size() - 1);

        if (!(layer instanceof OutputLayer outputLayer)) {
            throw new IllegalArgumentException("The last layer should be an instance of OutputLayer!");
        }

        return outputLayer;
    }
}
