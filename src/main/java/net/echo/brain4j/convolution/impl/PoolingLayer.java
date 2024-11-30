package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.pooling.PoolingType;

public class PoolingLayer extends Layer {

    protected final PoolingType poolingType;
    protected final int kernelWidth;
    protected final int kernelHeight;
    protected int stride;
    protected int padding;

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight) {
        super(kernelWidth * kernelHeight, Activations.LINEAR);
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.poolingType = poolingType;
    }

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight, int stride, int padding) {
        this(poolingType, kernelWidth, kernelHeight);
        this.stride = stride;
        this.padding = padding;
    }
}
