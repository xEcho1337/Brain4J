package com.github.echo.network.structure.layer;

import com.github.echo.types.Activations;
import com.github.echo.types.lost.LossFunctions;

public class OutputLayer extends DenseLayer {

    private final LossFunctions lossFunction;

    public OutputLayer(int neuronsCount, Activations activationFunction, LossFunctions lossFunction) {
        super(neuronsCount, activationFunction);
        this.lossFunction = lossFunction;
    }

    public LossFunctions getLossFunction() {
        return lossFunction;
    }
}
