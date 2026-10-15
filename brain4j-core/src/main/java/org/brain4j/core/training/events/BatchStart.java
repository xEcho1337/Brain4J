package org.brain4j.core.training.events;

import org.brain4j.core.training.Trainer;

/**
 * Signals the start of one mini-batch.
 * Use it to capture batch timing or prepare state.
 */
public record BatchStart(Trainer trainer, int batch, int totalBatches) implements TrainingEvent {}
