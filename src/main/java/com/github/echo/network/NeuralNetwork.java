package com.github.echo.network;

import com.github.echo.network.structure.layer.DenseLayer;
import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;
import com.github.echo.network.structure.layer.OutputLayer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {

    private final NeuralConfiguration configuration;
    private final List<DenseLayer> layers = new ArrayList<>();

    public NeuralNetwork(NeuralConfiguration configuration) {
        this.configuration = configuration;
        this.layers.addAll(configuration.getLayers());
        this.resetSynapses();
    }

    public NeuralNetwork(NeuralNetwork parent) {
        this.configuration = parent.getConfiguration();

        for (DenseLayer layer : parent.getLayers()) {
            configuration.layer(layer);
        }
    }

    /**
     * Gets the configuration of this neural network.
     * @return the neural configuration
     */
    public NeuralConfiguration getConfiguration() {
        return configuration;
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
            layer.getSynapses().clear();

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

        if (inputLayer.getNeurons().size() != input.length) {
            throw new IllegalArgumentException("Input layer size does not match input size");
        }

        // Reset every neuron value
        for (DenseLayer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
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

            for (Synapse synapse : layer.getSynapses()) {
                Neuron inputNeuron = synapse.getInputNeuron();
                Neuron outputNeuron = synapse.getOutputNeuron();

                // Weighted sum
                outputNeuron.setValue(outputNeuron.getValue() + inputNeuron.getValue() * synapse.getWeight());
            }

            // Apply the activation function
            for (Neuron neuron : nextLayer.getNeurons()) {
                neuron.applyFunction();
            }
        }

        DenseLayer outputLayer = layers.get(layers.size() - 1);
        double[] output = new double[outputLayer.getNeurons().size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeuronAt(i).getValue();
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

        DenseLayer layer = clone.getLayers().get(index);
        DenseLayer previous = clone.getLayers().get(index - 1);

        if (previous.getNeurons().size() != inputs.length) {
            throw new IllegalArgumentException("Input layer size does not match input size");
        }

        double[] outputs = new double[layer.getNeurons().size()];

        for (int i = 0; i < previous.getNeurons().size(); i++) {
            Neuron neuron = previous.getNeuronAt(i);
            neuron.setValue(inputs[i]);
        }

        for (int i = 0; i < layer.getNeurons().size(); i++) {
            Neuron outputNeuron = layer.getNeuronAt(i);

            for (Synapse synapse : layer.getSynapses()) {
                Neuron inputNeuron = synapse.getInputNeuron();
                Neuron neuron = synapse.getOutputNeuron();

                if (neuron.equals(outputNeuron)) {
                    outputNeuron.setValue(outputNeuron.getValue() + inputNeuron.getValue() * synapse.getWeight());
                }
            }

            outputNeuron.applyFunction();
            outputs[i] = outputNeuron.getValue();
        }

        return outputs;
    }

    /**
     * Returns the list of layers in the neural network.
     *
     * @return a list of Layer objects representing the layers in the neural network
     */
    public List<DenseLayer> getLayers() {
        return configuration.getLayers();
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
        return getLayers().get(index);
    }

    /**
     * Returns the last layer in the neural network.
     *
     * @return the output and last layer of the neural network.
     */
    public OutputLayer getOutputLayer() {
        DenseLayer layer = getLayers().get(getLayers().size() - 1);

        if (!(layer instanceof OutputLayer outputLayer)) {
            throw new IllegalArgumentException("The last layer should be an instance of OutputLayer!");
        }

        return outputLayer;
    }
}
