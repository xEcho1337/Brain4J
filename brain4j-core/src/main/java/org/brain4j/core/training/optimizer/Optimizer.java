package org.brain4j.core.training.optimizer;

import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.optimizer.impl.GradientDescent;
import org.brain4j.core.training.optimizer.impl.Lion;
import org.brain4j.math.tensor.Tensor;

/**
 * Computes parameter updates from gradients.
 * Use it to control convergence speed and stability during training.
 *
 * @see GradientDescent
 * @see Adam
 * @see AdamW
 * @see Lion
 */
public abstract class Optimizer {

    private double learningRate;

    /**
     * Creates an optimizer with the given learning rate.
     */
    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Applies one optimization step using the stored gradient.
     *
     * @return the optimized weights
     */
    public Tensor step(Tensor weights) {
        return step(weights, weights.grad());
    }

    /**
     * Applies one optimization step using the given gradient.
     *
     * @return the optimized weights
     */
    public abstract Tensor step(Tensor weights, Tensor gradient);

    /**
     * Initializes internal optimizer state.
     */
    public void initialize() {
    }

    /**
     * Runs a hook after each training batch.
     */
    public void postBatch() {
    }

    /**
     * Returns the learning rate.
     *
     * @return the current learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Replaces the learning rate.
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}
