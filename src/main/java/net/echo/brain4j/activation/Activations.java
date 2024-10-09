package net.echo.brain4j.activation;

import net.echo.brain4j.activation.impl.LinearActivation;
import net.echo.brain4j.activation.impl.ReLUActivation;
import net.echo.brain4j.activation.impl.SigmoidActivation;

public enum Activations {

    LINEAR(new LinearActivation()),
    RELU(new ReLUActivation()),
    SIGMOID(new SigmoidActivation());

    private final Activation function;

    Activations(Activation function) {
        this.function = function;
    }

    public Activation getFunction() {
        return function;
    }
}
