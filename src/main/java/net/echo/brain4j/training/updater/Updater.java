package net.echo.brain4j.training.updater;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Synapse;

import java.util.List;

public abstract class Updater {

    public void postInitialize() {
    }

    public void postIteration(List<Layer> layers, double learningRate) {
    }

    public void postFit(List<Layer> layers, double learningRate) {
    }

    public abstract void acknowledgeChange(Synapse synapse, double change, double learningRate);
}
