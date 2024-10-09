package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;

public class CrossEntropy implements LossFunction {

    @Override
    public double calculate(double[] actual, double[] predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.length; i++) {
            loss -= actual[i] * Math.log(predicted[i] + 1e-12);
        }

        return loss / actual.length;
    }
}
