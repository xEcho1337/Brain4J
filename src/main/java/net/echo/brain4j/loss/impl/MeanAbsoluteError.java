package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;

public class MeanAbsoluteError implements LossFunction {

    @Override
    public double calculate(double[] actual, double[] predicted) {
        double error = 0.0;

        for (int i = 0; i < actual.length; i++) {
            error += Math.abs(actual[i] - predicted[i]);
        }

        return error / actual.length;
    }
}
