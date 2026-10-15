package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInit;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

/**
 * Smooth ReLU approximation. Use it when a differentiable positive output helps.
 */
public class SoftPlus implements Activation {

    /**
     * {@inheritDoc}
     */
    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double activate(double input) {
        if (input > 20) {
            // Avoid overflow: log(1 + exp(x)) saturates to x.
            return input;
        } else if (input < -20) {
            // Avoid underflow: log(1 + exp(x)) saturates to exp(x).
            return Math.exp(input);
        }

        return Math.log1p(Math.exp(input));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double input) {
        if (input >= 0) {
            double expNeg = Math.exp(-input);

            return 1.0 / (1.0 + expNeg);
        } else {
            double expPos = Math.exp(input);

            return expPos / (1.0 + expPos);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getKernelPrefix() {
        return "softplus";
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getActivationId() {
        return 6;
    }
}
