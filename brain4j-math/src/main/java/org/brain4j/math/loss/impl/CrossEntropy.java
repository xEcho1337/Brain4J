package org.brain4j.math.loss.impl;

import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

/**
 * Categorical cross-entropy with optional class weights. Use it for multi-class classification.
 */
public record CrossEntropy(Tensor classWeights) implements LossFunction {

    /**
     * Creates an unweighted cross-entropy.
     */
    public CrossEntropy() {
        this(null);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;
        int numClasses = actual.shapeAt(actual.rank() - 1);

        for (int i = 0; i < actual.elements(); i++) {
            double y = actual.get(i);
            double p = predicted.get(i);
            double w = 1.0;

            if (classWeights != null) {
                int cls = i % numClasses;
                w = classWeights.get(cls);
            }

            loss -= w * y * Math.log(p + 1e-15);
        }

        int batchSize = actual.rank() > 1 ? actual.shapeAt(0) : 1;

        return loss / batchSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor delta(Tensor output, Tensor target, Tensor derivative) {
        Tensor error = output.minus(target);

        return classWeights == null ? error : error.mul(classWeights);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isRegression() {
        return false;
    }
}
