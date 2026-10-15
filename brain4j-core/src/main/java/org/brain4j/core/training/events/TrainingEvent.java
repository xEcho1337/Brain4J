package org.brain4j.core.training.events;

/**
 * Marks lifecycle events emitted during training.
 * Use it to observe batch, epoch and run boundaries.
 */
public sealed interface TrainingEvent permits BatchEnd, BatchStart, EpochEnd, EpochStart, TrainingEnd {}
