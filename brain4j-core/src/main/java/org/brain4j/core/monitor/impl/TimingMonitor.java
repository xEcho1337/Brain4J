package org.brain4j.core.monitor.impl;

import java.util.ArrayList;
import java.util.List;

import org.brain4j.core.monitor.Monitor;
import org.brain4j.core.training.Trainer;
import org.brain4j.core.training.events.BatchEnd;
import org.brain4j.core.training.events.BatchStart;
import org.brain4j.core.training.events.EpochStart;
import org.brain4j.core.training.events.TrainingEvent;

/**
 * Tracks per-batch timings over a sliding window.
 * Use it to estimate throughput and remaining training time.
 */
public class TimingMonitor implements Monitor {

    protected final List<Double> times = new ArrayList<>();
    protected final int timeWindow;

    protected double batchStartTime;
    protected int currentEpoch;
    protected int totalEpochs;
    protected double averageTimeBatch;

    /**
     * Create a monitor with the given window size.
     */
    public TimingMonitor(int timeWindow) {
        this.timeWindow = timeWindow;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void onEvent(TrainingEvent event, Trainer trainer) {
        switch (event) {
            case BatchStart ignored -> this.batchStartTime = System.nanoTime();
            case BatchEnd(int batch, int totalBatches) -> batchCompleted();
            case EpochStart(int epoch, int targetEpochs) -> epochStarted(epoch, targetEpochs);
            default -> {}
        }
    }

    private void batchCompleted() {
        double took = System.nanoTime() - batchStartTime;
        times.add(took / 1e6);

        if (times.size() > timeWindow) {
            times.removeFirst();
        }

        this.averageTimeBatch = times.stream().reduce(Double::sum).orElse(0.0);
    }

    /**
     * Record the start of a new epoch.
     */
    public void epochStarted(int epoch, int total) {
        this.currentEpoch = epoch;
        this.totalEpochs = total;
    }

    /**
     * Return the average time per batch.
     *
     * @return the average milliseconds
     */
    public double averagePerBatch() {
        return totalTime() / times.size();
    }

    /**
     * Return the accumulated batch time.
     *
     * @return the total milliseconds
     */
    public double totalTime() {
        return averageTimeBatch;
    }

    /**
     * Return the current epoch.
     *
     * @return the epoch index
     */
    public int currentEpoch() {
        return currentEpoch;
    }

    /**
     * Return the total epoch count.
     *
     * @return the total epochs
     */
    public int totalEpochs() {
        return totalEpochs;
    }
}
