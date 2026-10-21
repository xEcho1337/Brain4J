package org.brain4j.core.training.optimizer.impl;

import org.brain4j.math.tensor.Tensor;

/**
 * Adds decoupled weight decay to Adam updates.
 * Use it when L2 regularization should not pollute gradients.
 */
public class AdamW extends Adam {

    private final double weightDecay;

    /**
     * Create an optimizer with default weight decay.
     */
    public AdamW(double learningRate) {
        this(learningRate, 0.001);
    }

    /**
     * Create an optimizer with the given weight decay.
     */
    public AdamW(double learningRate, double weightDecay) {
        this(learningRate, weightDecay, 0.9, 0.999, 1e-8);
    }

    /**
     * Create an optimizer with the given decay and Adam hyperparameters.
     */
    public AdamW(double learningRate, double weightDecay, double beta1, double beta2, double epsilon) {
        super(learningRate, beta1, beta2, epsilon);
        this.weightDecay = weightDecay;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor step(Tensor weights, Tensor gradient, int samples) {
        Tensor adamValue = super.step(weights, gradient, samples);
        Tensor weightDecayTerm = weights.times(learningRate * weightDecay);

        return adamValue.add(weightDecayTerm);
    }

    /**
     * Returns the weight decay coefficient.
     *
     * @return the decay value
     */
    public double weightDecay() {
        return weightDecay;
    }
}
