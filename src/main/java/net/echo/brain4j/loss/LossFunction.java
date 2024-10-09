package net.echo.brain4j.loss;

public interface LossFunction {
    
    double calculate(double[] actual, double[] predicted);
}
