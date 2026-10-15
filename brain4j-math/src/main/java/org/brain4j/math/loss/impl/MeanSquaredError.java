package org.brain4j.math.loss.impl;

import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

/**
 * Mean squared error. Use it as the default regression loss.
 */
public class MeanSquaredError implements LossFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        float[] actualData = actual.data();
        float[] predictedData = predicted.data();

        for (int i = 0; i < actual.elements(); i++) {
            loss += Math.pow(actualData[i] - predictedData[i], 2);
        }

        return loss / actual.shapeAt(0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor delta(Tensor output, Tensor target, Tensor derivative) {
        Tensor error = output.minus(target);

        if (derivative == null) {
            return error;
        }

        return error.mul(derivative);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isRegression() {
        return true;
    }
}
