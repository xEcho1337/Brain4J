package com.github.echo.network.structure.layer.impl;

import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;
import com.github.echo.network.structure.layer.Layer;

import java.util.List;

public class DropoutLayer implements Layer {

    private final double value;

    public DropoutLayer(double value) {
        this.value = value;
    }

    /**
     * Returns the value of the dropout layer.
     *
     * @return the value
     */
    public double getValue() {
        return value;
    }

    @Override
    public List<Neuron> getNeurons() {
        return List.of();
    }

    @Override
    public List<Synapse> getSynapses() {
        return List.of();
    }

    @Override
    public void createSynapses(Layer nextLayer) {

    }

    @Override
    public Neuron getNeuronAt(int index) {
        return null;
    }
}
