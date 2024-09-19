package com.github.echo.types.loss;

import com.github.echo.types.loss.impl.BinaryCrossEntropy;
import com.github.echo.types.loss.impl.MeanSquaredError;

public enum LossFunctions {

    MSE(new MeanSquaredError()),
    BCE(new BinaryCrossEntropy());

    private final LossFunction function;

    LossFunctions(LossFunction function) {
        this.function = function;
    }

    public LossFunction getFunction() {
        return function;
    }
}
