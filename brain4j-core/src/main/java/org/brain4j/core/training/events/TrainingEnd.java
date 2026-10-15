package org.brain4j.core.training.events;

/**
 * Signals the end of the whole training run.
 * Use it to flush final metrics or release resources.
 */
public record TrainingEnd() implements TrainingEvent {
}
