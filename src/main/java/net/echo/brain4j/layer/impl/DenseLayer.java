package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;

public class DenseLayer extends Layer {

    public DenseLayer(int input, Activations activation) {
        super(input, activation);
    }
}
