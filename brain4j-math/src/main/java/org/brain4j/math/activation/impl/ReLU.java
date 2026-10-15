package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.NormalHeInit;

/**
 * Rectified linear unit. Use it as the default hidden-layer activation.
 */
public class ReLU implements Activation {

    /**
     * {@inheritDoc}
     */
    @Override
    public WeightInit defaultWeightInit() {
        return new NormalHeInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : 0;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getKernelPrefix() {
        return "relu";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getActivationId() {
        return 0;
    }
}
