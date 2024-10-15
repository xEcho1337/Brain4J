package net.echo.brain4j.training.optimizers;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.structure.Synapse;

/**
 * Abstract class for optimization algorithms.
 */
@JsonAdapter(OptimizerAdapter.class)
public abstract class Optimizer {

    protected double learningRate;

    /**
     * Initializes the optimizer with a specified learning rate.
     *
     * @param learningRate the learning rate
     */
    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Gets the current learning rate.
     *
     * @return learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets a new learning rate.
     *
     * @param learningRate the new learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Updates the given synapse based on the optimization algorithm.
     *
     * @param synapse   the synapse to update
     * @param timestep  the current timestep
     */
    public abstract void update(Synapse synapse, int timestep);
}