package org.brain4j.math.weightsinit;

import java.util.random.RandomGenerator;

/**
 * Generates initial neural network weights.
 * Use it to pick bounds matching the layer activation.
 */
public interface WeightInit {

    /**
     * Returns the symmetric bound for the given fan-in and fan-out.
     *
     * @return the maximum/minimum bound
     */
    double getBound(int input, int output);

    /**
     * Generates one weight for the given fan-in and fan-out.
     *
     * @return a random weight within the init bounds
     */
    double generate(RandomGenerator generator, int input, int output);

    /**
     * Returns a uniform value within the given bounds.
     *
     * @return a value in {@code [min, max)}
     */
    default double randomBetween(RandomGenerator generator, double min, double max) {
        return generator.nextDouble() * (max - min) + min;
    }
}
