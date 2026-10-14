package org.brain4j.core.training;

import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.loss.LossFunction;

/**
 * Groups loss, optimizer and updater for a training run.
 * Use it to configure a {@link Trainer}.
 */
public record TrainingConfig(LossFunction loss, Optimizer optimizer, Updater updater) {

    /**
     * Creates a config with a stochastic updater.
     *
     * @return a new config
     */
    public static TrainingConfig of(LossFunction loss, Optimizer optimizer) {
        return new TrainingConfig(loss, optimizer, new StochasticUpdater());
    }

    public TrainingConfig {
        if (loss == null) {
            throw new IllegalArgumentException("Loss cannot be null!");
        }

        if (optimizer == null) {
            throw new IllegalArgumentException("Optimizer cannot be null!");
        }

        if (updater == null) {
            throw new IllegalArgumentException("Updater cannot be null!");
        }
    }
}
