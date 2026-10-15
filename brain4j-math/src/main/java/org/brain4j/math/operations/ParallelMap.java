package org.brain4j.math.operations;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import org.brain4j.math.commons.D2DFunction;

/**
 * Applies an element-wise function in parallel. Use it to transform large
 * float buffers with ForkJoin splitting.
 */
public class ParallelMap extends RecursiveAction {

    /**
     * Carries the function and buffer shared by map tasks.
     *
     * @param function element-wise function
     * @param data buffer transformed in place
     */
    public record MapParameters(D2DFunction function, float[] data) { }

    private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static final int PARALLEL_WORK_THRESHOLD = PARALLELISM;
    private static final int PARALLEL_COMPLEXITY_THRESHOLD = 1 << 12;
    private static final int SPLIT_COMPLEXITY_THRESHOLD = 1 << 10;
    
    private static boolean isOverParallelThreshold(int work) {
        return work > PARALLEL_WORK_THRESHOLD && work > PARALLEL_COMPLEXITY_THRESHOLD;
    }
    
    private static boolean isOverSplitThreshold(int work) {
        return work > SPLIT_COMPLEXITY_THRESHOLD;
    }

    private final MapParameters parameters;
    private final int start;
    private final int end;

    /**
     * Create a task covering an array range.
     *
     * @param parameters shared function and buffer
     * @param start first index, inclusive
     * @param end last index, exclusive
     */
    public ParallelMap(MapParameters parameters, int start, int end) {
        this.parameters = parameters;
        this.start = start;
        this.end = end;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    protected void compute() {
        int work = end - start;

        if (isOverSplitThreshold(work)) {
            int mid = (start + end) >>> 1;
            invokeAll(
                new ParallelMap(parameters, start, mid),
                new ParallelMap(parameters, mid, end)
            );
            return;
        }

        D2DFunction function = parameters.function();
        float[] data = parameters.data();

        mapSection(function, start, end, data);
    }

    /**
     * Apply a function to every element, using ForkJoin when large.
     *
     * @param function element-wise function
     * @param data buffer transformed in place
     */
    public static void map(D2DFunction function, float[] data) {
        int start = 0;
        int end = data.length;
        int work = end - start;

        if (!isOverParallelThreshold(work)) {
            mapSection(function, start, end, data);
            return;
        }

        int step = work / PARALLELISM;
        var params = new MapParameters(function, data);
        List<ParallelMap> actions = new ArrayList<>();

        for (int i = 0; i < PARALLELISM; i++) {
            int startIndex = i * step;
            int endIndex = (i == PARALLELISM - 1) ? work : Math.min(startIndex + step, work);

            if (startIndex < endIndex) {
                actions.add(new ParallelMap(params, startIndex, endIndex));
            }

        }

        ForkJoinTask.invokeAll(actions);
    }

    private static void mapSection(
        D2DFunction function,
        int start,
        int end,
        float[] data
    ) {
        for (int i = start; i < end; i++) {
            data[i] = (float) function.apply(data[i]);
        }

    }
}