package org.brain4j.math.weightsinit.impl;

import java.util.random.RandomGenerator;

import org.brain4j.math.weightsinit.WeightInit;

import static org.brain4j.math.Constants.SQRT3;

/**
 * LeCun uniform init. Use it for SELU or sigmoid layers preserving variance.
 */
public class LeCunInit implements WeightInit {

    /**
     * {@inheritDoc}
     */
    @Override
    public double getBound(int input, int output) {
        return SQRT3 / Math.sqrt(input);
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
