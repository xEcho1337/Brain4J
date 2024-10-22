package net.echo.brain4j.training;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;

import java.util.List;

public class BackPropagation {

    private final Model model;
    private final Optimizer optimizer;

    private int timestep = 0;

    public BackPropagation(Model model, Optimizer optimizer) {
        this.model = model;
        this.optimizer = optimizer;
    }

    public void iterate(DataSet dataSet, double learningRate) {
        for (DataRow row : dataSet.getDataRows()) {
            double[] inputs = row.inputs();
            double[] targets = row.outputs();

            double[] outputs = model.predict(inputs);

            backpropagate(targets, outputs, learningRate);
        }
    }

    private void backpropagate(double[] targets, double[] outputs, double learningRate) {
        List<Layer> layers = model.getLayers();
        initialDelta(layers, targets, outputs);

        // Hidden layers error and delta
        for (int l = layers.size() - 2; l > 0; l--) {
            Layer layer = layers.get(l);

            if (layer instanceof DropoutLayer dropoutLayer) {
                Layer previous = layers.get(l - 1);
                dropoutLayer.backward(previous.getNeurons());
                continue;
            }

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
        Layer outputLayer = layers.get(layers.size() - 1);

        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeuronAt(i);

            double output = outputs[i];
            double error = targets[i] - output;

            double delta = error * outputLayer.getActivation().getFunction().getDerivative(output);
            neuron.setDelta(delta);
        }
    }

    private void updateWeightsAndBiases(List<Layer> layers, double learningRate) {
        timestep++;

        layers.parallelStream().forEach(nextLayer -> {
            for (Synapse synapse : nextLayer.getSynapses()) {
                optimizer.update(synapse, timestep);
            }

            for (Neuron neuron : nextLayer.getNeurons()) {
                double deltaBias = learningRate * neuron.getDelta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        });
    }
}