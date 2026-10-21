package org.brain4j.math.weightsinit.impl;

import java.util.random.RandomGenerator;

import org.brain4j.math.weightsinit.WeightInit;

/**
 * Xavier normal init. Use it for tanh or sigmoid layers balancing fan-in and fan-out.
 */
public class NormalXavierInit implements WeightInit {

    /**
     * {@inheritDoc}
     */
    @Override
    public double bound(int input, int output) {
        return 2.0 / (input + output);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        return randomBetween(generator, 0, bound(input, output));
    }
}
