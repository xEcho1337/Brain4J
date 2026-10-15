package org.brain4j.core.training.optimizer.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.training.optimizer.Optimizer;

/**
 * Applies plain gradient steps without momentum.
 * Use it as a baseline or when full control over updates is needed.
 */
public class GradientDescent extends Optimizer {

    protected GradientDescent() {
        super(0);
    }

    /**
     * Create an optimizer with the given learning rate.
     */
    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        return gradient;
    }
}
