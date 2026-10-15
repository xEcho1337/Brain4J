package org.brain4j.core.training.events;

/**
 * Signals the end of one mini-batch.
 * Use it to update batch-level metrics or timings.
 */
public record BatchEnd(int batch, int totalBatches) implements TrainingEvent {}
