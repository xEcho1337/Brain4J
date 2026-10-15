package org.brain4j.math.commons;

/**
 * Python-style index range. Use it for tensor slicing with negative-index support.
 */
public record Range(int start, int end, int step) {
    /**
     * Creates an interval range with unit step.
     *
     * @return the new range
     */
    public static Range interval(int start, int end) {
        return new Range(start, end);
    }

    /**
     * Creates a full-dimension range.
     *
     * @return the new range
     */
    public static Range all() {
        return new Range(0, Integer.MAX_VALUE, 1);
    }

    /**
     * Creates a single-index range.
     *
     * @return the new range
     */
    public static Range point(int i) {
        return new Range(i, i + 1, 1);
    }

    /**
     * Creates an interval range with unit step.
     */
    public Range(int start, int end) {
        this(start, end, 1);
    }

    /**
     * Creates a range with the given bounds and step.
     */
    public Range {
    }

    /**
     * Resolves the start index for a dimension.
     *
     * @return the non-negative start index
     */
    public int start(int dimSize) {
        return start >= 0 ? start : start + dimSize;
    }

    /**
     * Resolves the end index for a dimension.
     *
     * @return the clamped non-negative end index
     */
    public int end(int dimSize) {
        return (end >= 0 && start >= 0) ? Math.min(end, dimSize) : end + dimSize;
    }

    /**
     * Counts the elements covered in a dimension.
     *
     * @return the element count
     */
    public int size(int dimSize) {
        int s = start(dimSize);
        int e = end(dimSize);

        return (e - s + step - 1) / step;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int start() {
        return start;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int end() {
        return end;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int step() {
        return step;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return start + ":" + end + ":" + step;
    }
}
