package net.echo.brain4j.structure;

import com.google.gson.annotations.Expose;

import java.util.ArrayList;
import java.util.List;

public class Neuron {

    private final List<Synapse> synapses = new ArrayList<>();
    private final ThreadLocal<Double> localValue = new ThreadLocal<>();
    private final ThreadLocal<Double> delta = new ThreadLocal<>();
    private double totalDelta;
    @Expose private double bias = 2 * Math.random() - 1;

    public List<Synapse> getSynapses() {
        return synapses;
    }

    public void addSynapse(Synapse synapse) {
        this.synapses.add(synapse);
    }

    public void setTotalDelta(double totalDelta) {
        this.totalDelta = totalDelta;
    }

    public double getTotalDelta() {
        return totalDelta;
    }

    public double getDelta() {
        return delta.get();
    }

    public void setDelta(double delta) {
        this.delta.set(delta);
        this.totalDelta += delta;
    }

    public double getValue() {
        return localValue.get();
    }

    public void setValue(double value) {
        this.localValue.set(value);
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
