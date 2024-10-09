package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;

public class DropoutLayer extends Layer {

    private double dropout;

    public DropoutLayer() {
        super(0, Activations.LINEAR);
    }
}
