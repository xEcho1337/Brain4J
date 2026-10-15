package org.brain4j.core.training.updater.impl;

import org.brain4j.core.training.updater.Updater;

/**
 * Applies accumulated gradients once per epoch.
 * Use it for full-batch training with stable updates.
 */
public class NormalUpdater extends Updater {

    /**
     * {@inheritDoc}
     */
    @Override
    public void postFit(double learningRate, int samples) {
        updateWeights(learningRate, samples);
        initialize();
    }
}
