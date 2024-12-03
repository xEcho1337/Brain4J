package net.echo.brain4j.training;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public class BackPropagation {

    private final Model model;
    private final Optimizer optimizer;

    public BackPropagation(Model model, Optimizer optimizer) {
        this.model = model;
        this.optimizer = optimizer;
    }


    public void iterate(DataSet dataSet) {
        for (DataRow row : dataSet.getDataRows()) {
            Vector output = model.predict(row.inputs());
            Vector target = row.outputs();

            backpropagate(target.toArray(), output.toArray());
        }

        List<Layer> layers = model.getLayers();
        optimizer.postFit(layers);
    }

    public void backpropagate(double[] targets, double[] outputs) {
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
                for (Synapse synapse : neuron.getSynapses()) {
                    optimizer.applyGradientStep(layer, neuron, synapse);
                }
            }
        }

        optimizer.postIteration(layers);
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
}