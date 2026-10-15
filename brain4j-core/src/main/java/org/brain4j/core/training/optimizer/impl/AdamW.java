package org.brain4j.core.training.optimizer.impl;

import org.brain4j.math.tensor.Tensor;

/**
 * Adds decoupled weight decay to Adam updates.
 * Use it when L2 regularization should not pollute gradients.
 */
public class AdamW extends Adam {

    private double weightDecay;

    protected AdamW() {
    }

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
    public Tensor step(Tensor weights, Tensor gradient) {
        Tensor adamValue = super.step(weights, gradient);
        Tensor weightDecayTerm = weights.times(weightDecay);

        return adamValue.add(weightDecayTerm);
    }

    /**
     * Return the weight decay coefficient.
     *
     * @return the decay value
     */
    public double getWeightDecay() {
        return weightDecay;
    }

    /**
     * Replace the weight decay coefficient.
     */
    public void setWeightDecay(double weightDecay) {
        this.weightDecay = weightDecay;
    }
}
