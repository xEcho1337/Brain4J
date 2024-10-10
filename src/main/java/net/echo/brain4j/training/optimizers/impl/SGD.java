package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;

public class SGD extends Optimizer {

    public SGD(double learningRate) {
        super(learningRate);
    }

    @Override
    public void update(Synapse synapse, int timestep) {
        double deltaWeight = learningRate * synapse.getOutputNeuron().getDelta() * synapse.getInputNeuron().getValue();
        synapse.setWeight(synapse.getWeight() + deltaWeight);
    }
}
