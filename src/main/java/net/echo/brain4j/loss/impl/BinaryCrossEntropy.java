package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;

public class BinaryCrossEntropy implements LossFunction {

    @Override
    public double calculate(double[] actual, double[] predicted) {
        double error = 0.0;

        for (int i = 0; i < actual.length; i++) {
            double p = Math.max(Math.min(predicted[i], 1 - 1e-15), 1e-15);
            error += -actual[i] * Math.log(p) - (1 - actual[i]) * Math.log(1 - p);
        }

        return error / actual.length;
    }
}
