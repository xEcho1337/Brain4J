package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;

public class CategoricalCrossEntropy implements LossFunction {
    @Override
    public double calculate(double[] expected, double[] actual) {
        double sum = 0.0;
        for (int i = 0; i < expected.length; i++) {
            sum += -expected[i] * Math.log(actual[i] + 1e-15);
        }
        return sum;
    }
}
