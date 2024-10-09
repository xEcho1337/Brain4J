package net.echo.brain4j.model.impl;

import net.echo.brain4j.data.DataSet;
import net.echo.brain4j.initialization.InitializationType;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.BackPropagation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FeedForwardModel implements Model {

    private final List<Layer> layers;
    private BackPropagation propagation;
    private LossFunction function;

    public FeedForwardModel(Layer... layers) {
        this.layers = new ArrayList<>(Arrays.asList(layers));
    }

    @Override
    public void compile(InitializationType type, LossFunctions function) {
        this.function = function.getFunction();
        this.propagation = new BackPropagation(this);

        // Ignore the output layer
        for (int i = 0; i < layers.size() - 1; i++) {
            Layer layer = layers.get(i);
            Layer nextLayer = layers.get(i + 1);

            int nIn = layer.getNeurons().size();
            int nOut = nextLayer.getNeurons().size();

            double bound = type.getInitializer().getBound(nIn, nOut);

            layer.connectAll(nextLayer, bound);
        }
    }

    @Override
    public double fit(DataSet set, double learningRate) {;
        return propagation.iterate(set, learningRate);
    }

    @Override
    public double[] predict(double ... input) {
        Layer inputLayer = layers.getFirst();

        if (input.length != inputLayer.getNeurons().size()) {
            throw new IllegalArgumentException("Input size does not match model's input dimension!");
        }

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.setValue(0);
            }
        }

        for (int i = 0; i < input.length; i++) {
            inputLayer.getNeuronAt(i).setValue(input[i]);
        }

        for (int l = 0; l < layers.size(); l++) {
            Layer layer = layers.get(l);

            // We reached the output layer, which already has everything calculated
            if (l + 1 == layers.size()) break;

            Layer nextLayer = layers.get(l + 1);

            for (Synapse synapse : layer.getSynapses()) {
                Neuron inputNeuron = synapse.getInputNeuron();
                Neuron outputNeuron = synapse.getOutputNeuron();

                // Weighted sum
                outputNeuron.setValue(outputNeuron.getValue() + inputNeuron.getValue() * synapse.getWeight());
            }

            // Apply the activation function
            nextLayer.applyFunction();
        }

        Layer outputLayer = layers.getLast();
        double[] output = new double[outputLayer.getNeurons().size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeuronAt(i).getValue();
        }

        return output;
    }

    @Override
    public LossFunction getLossFunction() {
        return function;
    }

    @Override
    public List<Layer> getLayers() {
        return layers;
    }

    @Override
    public String getStats() {
        StringBuilder stats = new StringBuilder();
        stats.append(String.format("%-7s %-15s %-10s %-12s\n", "Index", "Layer name", "nIn, nOut", "TotalParams"));
        stats.append("================================================\n");

        int params = 0;

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            String layerType = layer.getClass().getSimpleName();
            int nIn = (i == 0) ? layer.getNeurons().size() : layers.get(i - 1).getNeurons().size();
            int nOut = layer.getNeurons().size();

            int totalParams = layer.getTotalParams();

            stats.append(String.format("%-7d %-15s %-10s %-12d\n",
                    i, layerType, nIn + ", " + nOut, totalParams));

            params += totalParams;
        }

        stats.append("================================================\n");
        stats.append("Total parameters: ").append(params).append("\n");
        stats.append("================================================\n");
        return stats.toString();
    }
}
