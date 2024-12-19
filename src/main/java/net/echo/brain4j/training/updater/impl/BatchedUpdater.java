package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;

import java.util.List;

public class BatchedUpdater extends NormalUpdater {

    @Override
    public void postFit(List<Layer> layers, double learningRate) {
    }

    @Override
    public void postBatch(List<Layer> layers, double learningRate) {
        super.postFit(layers, learningRate);
    }
}
