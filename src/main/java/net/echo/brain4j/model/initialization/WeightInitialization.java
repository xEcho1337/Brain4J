package net.echo.brain4j.model.initialization;

import net.echo.brain4j.model.initialization.impl.HeInit;
import net.echo.brain4j.model.initialization.impl.LeCunInit;
import net.echo.brain4j.model.initialization.impl.NormalInit;
import net.echo.brain4j.model.initialization.impl.XavierInit;

/**
 * Enum that defines the different types of weight initialization strategies used for neural networks.
 * Each type corresponds to a different algorithm that initializes the weights of a neural network layer.
 */
public enum WeightInitialization {

    /**
     * Uses a normal distribution for weight initialization.
     * Suitable for shallow networks or simple architectures.
     * May cause issues like vanishing or exploding gradients in deep networks.
     */
    NORMAL(new NormalInit()),

    /**
     * He Initialization (also known as Kaiming or MSRA Initialization):
     * Initializes weights with variance 2 / n_in, optimized for ReLU and its variants.
     * Ideal for deep networks with ReLU activation to prevent the vanishing gradient problem.
     */
    HE(new HeInit()),

    /**
     * Also known as Glorot Initialization: Initializes weights with variance 1 / (n_in + n_out).
     * Best suited for networks with symmetric activations like Tanh or Sigmoid.
     * Ensures proper scaling of activations and gradients, particularly in moderate-depth networks.
     */
    XAVIER(new XavierInit()),

    /**
     * Initializes weights with variance 1 / n_in, particularly for networks with Sigmoid activation.
     * Works well for shallow networks but may not perform optimally for deeper or ReLU-based networks.
     */
    LECUN(new LeCunInit());

    private final WeightInitializer initializer;

    /**
     * Constructor for the enum. Associates each type with a corresponding weight initializer.
     *
     * @param initializer the weight initializer for the corresponding type
     */
    WeightInitialization(WeightInitializer initializer) {
        this.initializer = initializer;
    }

    /**
     * Gets the weight initializer associated with the initialization type.
     *
     * @return the WeightInitializer instance for the current initialization type
     */
    public WeightInitializer getInitializer() {
        return initializer;
    }
}