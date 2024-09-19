package com.github.echo.network.structure.layer.impl;

import com.github.echo.types.Activations;
import com.github.echo.types.loss.LossFunctions;

public class OutputLayer extends DenseLayer {

    private final LossFunctions lossFunction;

    public OutputLayer(int neuronsCount, Activations activationFunction, LossFunctions lossFunction) {
        super(neuronsCount, activationFunction);
        this.lossFunction = lossFunction;
    }

    /**
     * Gets the loss function of the output layer.
     *
     * @return the loss function.
     */
    public LossFunctions getLossFunction() {
        return lossFunction;
    }
}
