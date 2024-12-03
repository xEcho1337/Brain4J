package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;

import java.util.List;

public class StochasticGD extends Optimizer {

    public StochasticGD(double learningRate) {
        super(learningRate);
    }

    @Override
    public void update(Synapse synapse) {
        double deltaWeight = learningRate * synapse.getOutputNeuron().getDelta() * synapse.getInputNeuron().getValue();
        synapse.setWeight(synapse.getWeight() + deltaWeight);
    }

    @Override
    public void postIteration(List<Layer> layers) {
        for (Layer layer : layers) {
            // 30% improvement using parallel stream. TODO: Implement GPU support for better parallelization
            layer.getSynapses().parallelStream().forEach(this::update);

            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getDelta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        }
    }
}
