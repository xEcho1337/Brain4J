package net.echo.brain4j.structure;

import com.google.gson.annotations.Expose;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    private final List<Synapse> synapses = new ArrayList<>();
    private ThreadLocal<Double> localValue = new ThreadLocal<>();
    private double delta;
    private double value;
    @Expose private double bias = 2 * Math.random() - 1;

    public List<Synapse> getSynapses() {
        return synapses;
    }

    public void addSynapse(Synapse synapse) {
        this.synapses.add(synapse);
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getLocalValue() {
        return localValue.get();
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        localValue.set(value);
        this.value = value;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
