package com.github.echo.training.impl;

import com.github.echo.network.NeuralNetwork;
import com.github.echo.network.structure.layer.DenseLayer;
import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;
import com.github.echo.training.DataRow;
import com.github.echo.training.DataSet;
import com.github.echo.training.AbstractModel;
import com.github.echo.types.loss.LossFunction;

import java.util.List;

public class Backpropagation extends AbstractModel {

    private final LossFunction function;

    public Backpropagation(NeuralNetwork neuralNetwork) {
        super(neuralNetwork);

        this.function = neuralNetwork.getOutputLayer().getLossFunction().getFunction();
    }

    @Override
    public void train(DataSet dataSet, double precision) {
        network.resetSynapses();

        do {
            iterate(dataSet);
        } while (!(totalError < precision));
    }

    @Override
    public void iterate(DataSet dataSet) {
        totalError = 0.0;

        for (DataRow row : dataSet.getRows()) {
            double[] inputs = row.inputs();
            double[] targets = row.outputs();

            double[] outputs = network().calculateOutput(inputs);
            totalError += function.compute(targets, outputs);

            backpropagate(targets, outputs);
        }

        totalError /= dataSet.getRows().size();
    }

    private void backpropagate(double[] targets, double[] outputs) {
        List<DenseLayer> layers = network().getLayers();

        // Output layer error and delta
        DenseLayer outputLayer = layers.get(layers.size() - 1);

        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeuronAt(i);
            double output = outputs[i];

            // Calculate the error
            double error = targets[i] - output;
            double delta = error * neuron.getActivationFunction().derivative().apply(output);

            neuron.setDelta(delta);
        }

        // Hidden layers error and delta
        for (int l = layers.size() - 2; l > 0; l--) {
            DenseLayer layer = layers.get(l);
            DenseLayer nextLayer = layers.get(l + 1);

            for (Neuron neuron : layer.getNeurons()) {
                double output = neuron.getValue();
                double error = 0.0;

                for (Synapse synapse : nextLayer.getSynapses()) {
                    for (Neuron nextNeuron : nextLayer.getNeurons()) {
                        if (!synapse.getInputNeuron().equals(neuron)) continue;

                        error += synapse.getWeight() * nextNeuron.delta();
                    }
                }

                double delta = error * neuron.getActivationFunction().derivative().apply(output);
                neuron.setDelta(delta);
            }
        }

        // Update weights and biases
        for (int l = 0; l < layers.size() - 1; l++) {
            DenseLayer nextLayer = layers.get(l + 1);

            for (Synapse synapse : nextLayer.getSynapses()) {
                double deltaWeight = learningRate * synapse.getOutputNeuron().delta() * synapse.getInputNeuron().getValue();
                synapse.setWeight(synapse.getWeight() + deltaWeight);
            }

            for (Neuron neuron : nextLayer.getNeurons()) {
                double deltaBias = learningRate * neuron.delta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        }
    }
}