package com.github.echo.types.loss;

public interface LossFunction {

    double compute(double[] expected, double[] actual);
}
