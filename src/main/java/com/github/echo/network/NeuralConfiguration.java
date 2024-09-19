package com.github.echo.network;

import com.github.echo.network.structure.layer.Layer;
import com.github.echo.network.structure.layer.impl.DenseLayer;

import java.util.ArrayList;
import java.util.List;

public class NeuralConfiguration {

    private final List<Layer> layers = new ArrayList<>();

    /**
     * Adds a layer to the neural network and returns the updated NeuralNetwork object.
     *
     * @param  layer  the layer to be added to the neural network
     * @return        the updated NeuralNetwork object with the added layer
     */
    public NeuralConfiguration layer(Layer layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Returns the list of layers in the neural network.
     *
     * @return a list of Layer objects representing the layers in the neural network
     */
    public List<Layer> getLayers() {
        return layers;
    }
}
