package net.echo.brain4j.loss;

import net.echo.brain4j.loss.impl.BinaryCrossEntropy;
import net.echo.brain4j.loss.impl.CategoricalCrossEntropy;
import net.echo.brain4j.loss.impl.CrossEntropy;
import net.echo.brain4j.loss.impl.MeanSquaredError;

public enum LossFunctions {

    /**
     * Mean Squared Error (MSE): Used to evaluate the error in regression tasks by calculating
     * the average of the squared differences between predicted and actual values.
     * It is sensitive to outliers.
     */
    MEAN_SQUARED_ERROR(new MeanSquaredError()),

    /**
     * Binary Cross Entropy: Used to evaluate the error in binary classification tasks
     * by measuring the divergence between the predicted probabilities and the actual binary labels.
     * Suitable for models with a single output neuron using a sigmoid activation function.
     */
    BINARY_CROSS_ENTROPY(new BinaryCrossEntropy()),

    /**
     * Cross Entropy: Used to evaluate the error in multi-class classification tasks
     * by measuring the difference between the predicted probability distribution
     * and the actual distribution of classes. Typically used with models having multiple output neurons
     * and a softmax activation function.
     */
    CROSS_ENTROPY(new CrossEntropy()),

    /**
     * Categorical Cross Entropy: A variant of Cross Entropy specifically designed for multi-class classification
     * tasks. It is used when the target labels are one-hot encoded. It calculates the divergence between
     * the predicted probability distribution and the actual distribution of classes.
     */
    CATEGORICAL_CROSS_ENTROPY(new CategoricalCrossEntropy());

    private final LossFunction function;

    LossFunctions(LossFunction function) {
        this.function = function;
    }

    public LossFunction getFunction() {
        return function;
    }
}
