package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.updater.Updater;

import java.util.Arrays;
import java.util.List;

public class NormalUpdater extends Updater {

    private Synapse[] synapses;
    private Double[] gradients;

    @Override
    public void postInitialize() {
        this.synapses = new Synapse[Synapse.ID_COUNTER];
        this.gradients = new Double[Synapse.ID_COUNTER];

        Arrays.fill(gradients, 0.0);
    }

    @Override
    public void postFit(List<Layer> layers, double learningRate) {
        for (int i = 0; i < gradients.length; i++) {
            Synapse synapse = synapses[i];
            double gradient = gradients[i];

            synapse.setWeight(synapse.getWeight() + learningRate * gradient);
        }

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getDelta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        }

        Arrays.fill(gradients, 0.0);
    }

    @Override
    public void acknowledgeChange(Synapse synapse, double change, double learningRate) {
        int id = synapse.getSynapseId();

        synapses[id] = synapse;
        gradients[id] += change;
    }
}
