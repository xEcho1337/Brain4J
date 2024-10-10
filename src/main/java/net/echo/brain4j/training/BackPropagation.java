package net.echo.brain4j.training;

import net.echo.brain4j.data.DataRow;
import net.echo.brain4j.data.DataSet;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;

import java.util.List;

public class BackPropagation {

    private final Model model;
    private final LossFunction lossFunction;

    public BackPropagation(Model model) {
        this.model = model;
        this.lossFunction = model.getLossFunction();
    }
    public double iterate(DataSet dataSet, double learningRate) {
        double totalError = 0.0;

        for (DataRow row : dataSet.getDataRows()) {
            double[] inputs = row.inputs();
            double[] targets = row.outputs();

            double[] outputs = model.predict(inputs);

            totalError += lossFunction.calculate(targets, outputs);

            backpropagate(targets, outputs, learningRate);
        }

        return totalError / dataSet.getDataRows().size();
    }
    
    private void backpropagate(double[] targets, double[] outputs, double learningRate) {
        List<Layer> layers = model.getLayers();

        initialDelta(layers, targets, outputs);

        // Hidden layers error and delta
        for (int l = layers.size() - 2; l > 0; l--) {
            Layer layer = layers.get(l);

            for (Neuron neuron : layer.getNeurons()) {
                double output = neuron.getValue();
                double error = 0.0;

                Synapse synapse = neuron.getSynapse();
                error += synapse.getWeight() * synapse.getOutputNeuron().getDelta();

                double delta = error * layer.getActivation().getFunction().getDerivative(output);
                neuron.setDelta(delta);
            }
        }

        updateWeightsAndBiases(layers, learningRate);
    }

    private void initialDelta(List<Layer> layers, double[] targets, double[] outputs) {
        Layer outputLayer = layers.getLast();

        // Output layer error and delta
        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeuronAt(i);

            double output = outputs[i];
            double error = targets[i] - output;

            double delta = error * outputLayer.getActivation().getFunction().getDerivative(output);
            neuron.setDelta(delta);
        }
    }

    private void updateWeightsAndBiases(List<Layer> layers, double learningRate) {
        for (int l = 0; l < layers.size() - 1; l++) {
            Layer nextLayer = layers.get(l + 1);

            for (Synapse synapse : nextLayer.getSynapses()) {
                double deltaWeight = learningRate * synapse.getOutputNeuron().getDelta() * synapse.getInputNeuron().getValue();
                synapse.setWeight(synapse.getWeight() + deltaWeight);
            }

            for (Neuron neuron : nextLayer.getNeurons()) {
                double deltaBias = learningRate * neuron.getDelta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        }
    }
}
