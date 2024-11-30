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

    private static final double GRADIENT_CLIP = 5.0;

    private final Model model;
    private final Optimizer optimizer;
    private int timestep = 0;

    public BackPropagation(Model model, Optimizer optimizer) {
        this.model = model;
        this.optimizer = optimizer;
    }

    private double clipGradient(double gradient) {
        return Math.max(Math.min(gradient, GRADIENT_CLIP), -GRADIENT_CLIP);
    }

    public void iterate(DataSet dataSet, double learningRate) {
        for (DataRow row : dataSet.getDataRows()) {
            double[] inputs = row.inputs();
            double[] targets = row.outputs();

            double[] outputs = model.predict(inputs);

            backpropagate(targets, outputs, learningRate);
        }
    }

    public void backpropagate(double[] targets, double[] outputs, double learningRate) {
        List<Layer> layers = model.getLayers();
        initialDelta(layers, targets, outputs);

        for (int l = layers.size() - 2; l > 0; l--) {
            Layer layer = layers.get(l);

            if (layer instanceof DropoutLayer dropoutLayer) {
                Layer previous = layers.get(l - 1);
                dropoutLayer.backward(previous.getNeurons());
                continue;
            }

            for (Neuron neuron : layer.getNeurons()) {
                double output = neuron.getValue();

                for (Synapse synapse : neuron.getSynapses()) {
                    double error = clipGradient(synapse.getWeight() * synapse.getOutputNeuron().getDelta());
                    double delta = clipGradient(error * layer.getActivation().getFunction().getDerivative(output));

                    neuron.setDelta(delta);
                    synapse.setWeight(synapse.getWeight() + clipGradient(delta * synapse.getInputNeuron().getValue()));
                }
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

        for (Layer layer : layers) {
            // 30% improvement using parallel stream. TODO: Implement GPU support for better parallelization
            layer.getSynapses().parallelStream().forEach(synapse -> optimizer.update(synapse, timestep));

            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getDelta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        }
    }
}