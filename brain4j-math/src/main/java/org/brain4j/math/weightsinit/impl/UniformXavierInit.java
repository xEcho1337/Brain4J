package org.brain4j.math.weightsinit.impl;

import java.util.random.RandomGenerator;

import org.brain4j.math.weightsinit.WeightInit;

/**
 * Xavier uniform init. Use it for linear or tanh layers with uniform bounds.
 */
public class UniformXavierInit implements WeightInit {

    /**
     * {@inheritDoc}
     */
    @Override
    public double getBound(int input, int output) {
        return Math.sqrt(6.0 / (input + output));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        double bound = getBound(input, output);

        return randomBetween(generator, -bound, bound);
    }
}
