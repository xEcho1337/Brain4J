package net.echo.brain4j.structure;

import com.google.gson.annotations.Expose;

public class Neuron {

    private Synapse synapse;
    private double delta;
    private double value;
    @Expose private double bias = 2 * Math.random() - 1;

    public Synapse getSynapse() {
        return synapse;
    }

    public void setSynapse(Synapse synapse) {
        this.synapse = synapse;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
