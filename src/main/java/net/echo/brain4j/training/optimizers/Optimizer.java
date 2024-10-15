package net.echo.brain4j.training.optimizers;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.structure.Synapse;

@JsonAdapter(OptimizerAdapter.class)
public abstract class Optimizer {

    protected double learningRate;

    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public abstract void update(Synapse synapse, int timestep);
}
