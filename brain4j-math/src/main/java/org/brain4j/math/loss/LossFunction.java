package org.brain4j.math.loss;

import org.brain4j.math.loss.impl.BinaryCrossEntropy;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.math.loss.impl.MeanAbsoluteError;
import org.brain4j.math.loss.impl.MeanSquaredError;
import org.brain4j.math.tensor.Tensor;

/**
 * Measures prediction error and drives backpropagation.
 * Use it to score outputs during training.
 *
 * @see CrossEntropy
 * @see BinaryCrossEntropy
 * @see MeanAbsoluteError
 * @see MeanSquaredError
 */
public interface LossFunction {

    /**
     * Computes the error between expected and predicted outputs.
     *
     * @return a scalar error value
     */
    double calculate(Tensor expected, Tensor predicted);

    /**
     * Computes the output-layer delta for backpropagation.
     *
     * @return the delta tensor
     */
    Tensor delta(Tensor output, Tensor target, Tensor derivative);

    /**
     * Checks whether this loss targets regression tasks.
     *
     * @return true for regression, false for classification
     */
    boolean isRegression();
}
