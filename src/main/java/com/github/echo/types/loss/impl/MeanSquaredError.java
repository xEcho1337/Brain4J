package com.github.echo.types.loss.impl;

import com.github.echo.types.loss.LossFunction;

public class MeanSquaredError implements LossFunction {

    @Override
    public double compute(double[] expected, double[] actual) {
        double sum = 0;

        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i] - actual[i], 2);
        }

        return sum / expected.length;
    }
}