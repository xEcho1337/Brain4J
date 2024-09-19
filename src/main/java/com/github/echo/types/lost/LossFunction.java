package com.github.echo.types.lost;

public interface LossFunction {

    double compute(double[] expected, double[] actual);
}
