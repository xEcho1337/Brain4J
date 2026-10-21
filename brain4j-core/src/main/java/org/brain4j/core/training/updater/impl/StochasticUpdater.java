package org.brain4j.core.training.updater.impl;

import org.brain4j.core.training.updater.Updater;

/**
 * Applies accumulated gradients after every batch.
 * Use it for mini-batch stochastic training.
 */
public class StochasticUpdater extends Updater {

    /**
     * {@inheritDoc}
     */
    @Override
    public void postBatch() {
        updateWeights();
        initialize();
    }
}
