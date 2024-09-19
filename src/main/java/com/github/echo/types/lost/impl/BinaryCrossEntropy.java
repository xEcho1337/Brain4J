package com.github.echo.types.lost.impl;

import com.github.echo.types.lost.LossFunction;

public class BinaryCrossEntropy implements LossFunction {

    @Override
    public double compute(double[] expected, double[] actual) {
        double epsilon = 1e-15;  // Avoid log(0)
        double loss = 0.0;

        for (int i = 0; i < expected.length; i++) {
            double expectedValue = expected[i];
            double actualValue = actual[i];

            actualValue = Math.max(epsilon, Math.min(1 - epsilon, actualValue));

            // Binary Cross-Entropy formula
            loss += expectedValue * Math.log(actualValue) + (1 - expectedValue) * Math.log(1 - actualValue);
        }

        return -loss / expected.length;
    }
}
