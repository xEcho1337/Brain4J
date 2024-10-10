package net.echo.brain4j.training.optimizers;

import net.echo.brain4j.structure.Synapse;

public abstract class Optimizer {

    protected double learningRate;

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public abstract void update(Synapse synapse, int timestep);
}
