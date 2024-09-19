package com.github.echo.network.structure.layer;

import com.github.echo.network.structure.Neuron;
import com.github.echo.network.structure.Synapse;

import java.util.List;

public interface Layer {

    List<Neuron> getNeurons();

    List<Synapse> getSynapses();

    void createSynapses(Layer nextLayer);

    Neuron getNeuronAt(int index);
}
