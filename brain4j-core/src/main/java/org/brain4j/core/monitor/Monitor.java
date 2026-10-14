package org.brain4j.core.monitor;

import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.TrainingEvent;

/**
 * Observes training lifecycle events.
 * Use it to log progress, collect metrics or trigger side effects.
 */
public interface Monitor {
    /**
     * Handles a training event.
     */
    void onEvent(TrainingEvent event, Trainer trainer);
}
