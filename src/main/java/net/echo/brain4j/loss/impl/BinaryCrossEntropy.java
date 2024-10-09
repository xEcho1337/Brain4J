package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;

public class BinaryCrossEntropy implements LossFunction {

    @Override
    public double calculate(double[] actual, double[] predicted) {
        double error = 0.0;

        for (int i = 0; i < actual.length; i++) {
            error += -actual[i] * Math.log(predicted[i]) - (1 - actual[i]) * Math.log(1 - predicted[i]);
        }

        return error / actual.length;
    }
}
