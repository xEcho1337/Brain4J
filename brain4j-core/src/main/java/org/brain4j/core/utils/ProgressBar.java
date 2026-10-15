package org.brain4j.core.utils;

import java.util.Iterator;

import org.brain4j.math.commons.Commons;

/**
 * Wraps an iterable with console progress reporting.
 * Use it to display batch progress during training.
 *
 * @param <T> the iterated element type
 */
public class ProgressBar<T> implements Iterator<T>, Iterable<T> {

    private final Iterator<T> input;
    private final String taskName;
    private final long count;

    private long lastIteration;
    private long totalTime;
    private int current;

    /**
     * Create a bar for the given workload.
     */
    public ProgressBar(Iterable<T> input, String taskName) {
        this.input = input.iterator();
        this.taskName = taskName;
        this.count = input.spliterator().estimateSize();
        this.lastIteration = -1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean hasNext() {
        return input.hasNext();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public T next() {
        if (current == 0) {
            lastIteration = System.currentTimeMillis();
        }

        long diff = System.currentTimeMillis() - lastIteration;
        totalTime += diff;

        double percentage = (double) current / count;
        double averageTime = (double) totalTime / current;

        boolean asciiBar = Boolean.parseBoolean(System.getProperty("brain4j.ascii-progressbar"));

        String timeStr = Commons.formatDuration(averageTime / 1000);
        String barChar = asciiBar ? "=" : Commons.HEADER_CHAR;

        String progressBar = Commons.createProgressBar(
            percentage, 25,
            "<green>", barChar,
            "<reset>", barChar
        ) + " ";
        String progress = Colored.renderText(progressBar);

        String batches = Colored.renderText("<blue>%s<white>/<blue>%s", current + 1, count);
        String time = Colored.renderText("<gray> (%s/step)<reset>", timeStr);

        String message = taskName + progress + batches + time;
        System.out.print("\r" + message);

        lastIteration = System.currentTimeMillis();
        current += 1;

        return input.next();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Iterator<T> iterator() {
        return this;
    }
}
