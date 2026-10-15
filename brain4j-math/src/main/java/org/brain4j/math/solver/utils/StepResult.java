package org.brain4j.math.solver.utils;

import org.brain4j.math.tensor.Tensor;

/**
 * Adaptive solver step outcome. Use it to chain accepted states with retried steps.
 */
public record StepResult(Tensor nextHidden, Tensor nextDeltaT, boolean accepted) {

    /**
     * Creates an accepted step.
     *
     * @return the accepted result
     */
    public static StepResult accepted(Tensor nextHidden, Tensor nextDeltaT) {
        return new StepResult(nextHidden, nextDeltaT, true);
    }

    /**
     * Creates a rejected step.
     *
     * @return the rejected result
     */
    public static StepResult rejected(Tensor currentHidden, Tensor reducedDeltaT) {
        return new StepResult(currentHidden, reducedDeltaT, false);
    }
}
