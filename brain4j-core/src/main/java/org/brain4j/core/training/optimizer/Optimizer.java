package org.brain4j.core.training.optimizer;

import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.optimizer.impl.GradientDescent;
import org.brain4j.core.training.optimizer.impl.Lion;
import org.brain4j.math.tensor.Tensor;

/**
 * Computes parameter updates from gradients.
 * Implementations return the final update, including learning rate and sample normalization.
 *
 * @see GradientDescent
 * @see Adam
 * @see AdamW
 * @see Lion
 */
public interface Optimizer {

    /**
     * Computes an update using the stored gradient and a single sample.
     *
     * @return the update to subtract from the weights
     */
    default Tensor step(Tensor weights) {
        return step(weights, weights.grad(), 1);
    }

    /**
     * Computes an update using the given gradient and a single sample.
     *
     * @return the update to subtract from the weights
     */
    default Tensor step(Tensor weights, Tensor gradient) {
        return step(weights, gradient, 1);
    }

    /**
     * Computes an update using the given gradient.
     *
     * @param samples number of samples in the current batch (must be positive)
     * @return the update to subtract from the weights
     */
    Tensor step(Tensor weights, Tensor gradient, int samples);

    /**
     * Initializes internal optimizer state.
     */
    default void initialize() {
    }

    /**
     * Runs a hook after each training batch.
     */
    default void postBatch() {
    }

    /**
     * Returns the learning rate.
     *
     * @return the current learning rate
     */
    double learningRate();
}
