package org.brain4j.math.activation;

import org.brain4j.math.activation.impl.ELU;
import org.brain4j.math.activation.impl.GELU;
import org.brain4j.math.activation.impl.LeakyReLU;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.activation.impl.Mish;
import org.brain4j.math.activation.impl.ReLU;
import org.brain4j.math.activation.impl.Sigmoid;
import org.brain4j.math.activation.impl.SoftPlus;
import org.brain4j.math.activation.impl.Softmax;
import org.brain4j.math.activation.impl.Swish;
import org.brain4j.math.activation.impl.Tanh;

/**
 * Catalog of built-in activation functions.
 * Use it to pick a non-linearity by name.
 */
public enum Activations {

    /** Linear projection without non-linearity. */
    LINEAR(new Linear()),

    /**
     * Exponential Linear Unit, {@code alpha * (exp(x) - 1)} for negatives.
     */
    ELU(new ELU()),

    /**
     * Rectified Linear Unit, {@code max(0, x)}.
     */
    RELU(new ReLU()),

    /**
     * Gaussian Error Linear Unit approximation.
     */
    GELU(new GELU()),

    /**
     * Leaky ReLU with a small slope for negatives.
     */
    LEAKY_RELU(new LeakyReLU()),

    /**
     * Sigmoid mapping values to {@code [0, 1]}.
     */
    SIGMOID(new Sigmoid()),

    /**
     * Softmax normalizing outputs into a distribution.
     */
    SOFTMAX(new Softmax()),

    /**
     * Hyperbolic tangent mapping values to {@code [-1, 1]}.
     */
    TANH(new Tanh()),

    /**
     * Mish, {@code x * tanh(softplus(x))}.
     */
    MISH(new Mish()),

    /**
     * Swish, {@code x * sigmoid(x)}.
     */
    SWISH(new Swish()),

    /**
     * Softplus, {@code ln(1 + e^x)}.
     */
    SOFTPLUS(new SoftPlus());

    private final Activation function;

    Activations(Activation function) {
        this.function = function;
    }

    /**
     * Returns the activation function for this constant.
     *
     * @return the backing activation
     */
    public Activation function() {
        return function;
    }
}
