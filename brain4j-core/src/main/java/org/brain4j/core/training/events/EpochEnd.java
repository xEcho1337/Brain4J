package org.brain4j.core.training.events;

/**
 * Signals the end of one epoch.
 * Use it to run validation or log epoch metrics.
 */
public record EpochEnd(int epoch, int totalEpochs) implements TrainingEvent {}
