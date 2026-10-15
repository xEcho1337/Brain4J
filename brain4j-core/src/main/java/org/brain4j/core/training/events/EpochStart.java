package org.brain4j.core.training.events;

/**
 * Signals the start of one epoch.
 * Use it to reset epoch state or log progress.
 */
public record EpochStart(int epoch, int totalEpochs) implements TrainingEvent {}
