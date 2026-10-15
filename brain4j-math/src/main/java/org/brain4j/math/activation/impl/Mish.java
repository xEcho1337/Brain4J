package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.NormalXavierInit;

/**
 * Mish activation. Use it for deep vision models needing smooth non-monotonicity.
 */
public class Mish implements Activation {

    /**
     * {@inheritDoc}
     */
    @Override
    public WeightInit defaultWeightInit() {
        return new NormalXavierInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double activate(double input) {
        double softplus = Math.log1p(Math.exp(input));
        return input * Math.tanh(softplus);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        double softplus = Math.log1p(Math.exp(input));
        double tanhSp = Math.tanh(softplus);
        double sigmoid = 1.0 / (1.0 + Math.exp(-input));

        return tanhSp + input * sigmoid * (1 - tanhSp * tanhSp);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getKernelPrefix() {
        return "mish";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getActivationId() {
        return 8;
    }
}
