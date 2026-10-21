package org.brain4j.math.weightsinit.impl;

import java.util.random.RandomGenerator;

import org.brain4j.math.weightsinit.WeightInit;

/**
 * He normal init. Use it for ReLU layers needing variance-preserving weights.
 */
public class NormalHeInit implements WeightInit {

    /**
     * {@inheritDoc}
     */
    @Override
    public double bound(int input, int output) {
        return 2.0 / input;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        return randomBetween(generator, 0, bound(input, output));
    }
}
