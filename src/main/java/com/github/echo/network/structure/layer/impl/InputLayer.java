package com.github.echo.network.structure.layer.impl;

import com.github.echo.activations.Activations;
import com.github.echo.network.structure.layer.Layer;

public class InputLayer extends Layer {

    public InputLayer(int neuronsCount) {
        super(neuronsCount, Activations.LINEAR);
    }
}
