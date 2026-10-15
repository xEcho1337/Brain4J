package org.brain4j.math.weightsinit.impl;

import java.util.random.RandomGenerator;

import org.brain4j.math.weightsinit.WeightInit;

/**
 * Plain normal init. Use it for small-scale random baselines.
 */
public class NormalInit implements WeightInit {

    /**
     * {@inheritDoc}
     */
    @Override
    public double getBound(int input, int output) {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        return randomBetween(generator, -1, 1);
    }
}
