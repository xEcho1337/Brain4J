package org.brain4j.core.training.optimizer.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.training.optimizer.Optimizer;

/**
 * Applies plain gradient steps without momentum.
 * Use it as a baseline or when full control over updates is needed.
 */
public record GradientDescent(double learningRate) implements Optimizer {

    /**
     * Create an optimizer with the given learning rate.
     */
    public GradientDescent {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor step(Tensor weights, Tensor gradient, int samples) {
        return gradient.divide(samples).mul(learningRate);
    }
}
