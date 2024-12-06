package net.echo.brain4j.activation;

import net.echo.brain4j.activation.impl.*;

public enum Activations {

    LINEAR(new LinearActivation()),
    RELU(new ReLUActivation()),
    GELU(new GELUActivation()),
    LEAKY_RELU(new LeakyReLUActivation()),
    SIGMOID(new SigmoidActivation()),
    SOFTMAX(new SoftmaxActivation()),
    TANH(new TanhActivation());

    private final Activation function;

    Activations(Activation function) {
        this.function = function;
    }

    public Activation getFunction() {
        return function;
    }
}
