package net.echo.brain4j.loss;

import net.echo.brain4j.loss.impl.BinaryCrossEntropy;
import net.echo.brain4j.loss.impl.CrossEntropy;
import net.echo.brain4j.loss.impl.MeanSquaredError;

public enum LossFunctions {

    MEAN_SQUARED_ERROR(new MeanSquaredError()),
    BINARY_CROSS_ENTROPY(new BinaryCrossEntropy()),
    CROSS_ENTROPY(new CrossEntropy());

    private final LossFunction function;

    LossFunctions(LossFunction function) {
        this.function = function;
    }

    public LossFunction getFunction() {
        return function;
    }
}
