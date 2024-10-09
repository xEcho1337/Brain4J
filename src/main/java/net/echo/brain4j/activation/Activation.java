package net.echo.brain4j.activation;

public interface Activation {

    double activate(double input);

    double getDerivative(double input);
}
