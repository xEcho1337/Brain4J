package org.brain4j.math.loss.impl;

import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

/**
 * Mean absolute error. Use it for robust regression with outlier tolerance.
 */
public class MeanAbsoluteError implements LossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.elements(); i++) {
            loss += Math.abs(actual.get(i) - predicted.get(i));
        }

        return loss / actual.shapeAt(0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor delta(Tensor output, Tensor target, Tensor derivative) {
        Tensor error = output.minus(target);

        return error.map(Math::signum);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isRegression() {
        return true;
    }
}
