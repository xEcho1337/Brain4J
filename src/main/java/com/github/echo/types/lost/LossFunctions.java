package com.github.echo.types.lost;

import com.github.echo.types.lost.impl.BinaryCrossEntropy;
import com.github.echo.types.lost.impl.MeanSquaredError;

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
