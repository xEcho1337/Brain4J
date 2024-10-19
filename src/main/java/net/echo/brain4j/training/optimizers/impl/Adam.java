package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;

import java.util.HashMap;
import java.util.Map;

public class Adam extends Optimizer {

    // Momentum vectors
    private final Map<Synapse, Double> firstMomentum = new HashMap<>();
    private final Map<Synapse, Double> secondMomentum = new HashMap<>();

    private double beta1;
    private double beta2;
    private double epsilon;

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    @Override
    public void update(Synapse synapse, int timestep) {
        double gradient = synapse.getOutputNeuron().getDelta() * synapse.getInputNeuron().getValue();

        double currentFirstMomentum = firstMomentum.getOrDefault(synapse, 0.0);
        double currentSecondMomentum = secondMomentum.getOrDefault(synapse, 0.0);

        double m = beta1 * currentFirstMomentum + (1 - beta1) * gradient;
        double v = beta2 * currentSecondMomentum + (1 - beta2) * gradient * gradient;

        firstMomentum.put(synapse, m);
        secondMomentum.put(synapse, v);

        double mHat = m / (1 - Math.pow(beta1, timestep));
        double vHat = v / (1 - Math.pow(beta2, timestep));

        double deltaWeight = (learningRate * mHat) / (Math.sqrt(vHat) + epsilon);
        synapse.setWeight(synapse.getWeight() + deltaWeight);
    }

    public double getBeta1() {
        return beta1;
    }

    public void setBeta1(double beta1) {
        this.beta1 = beta1;
    }

    public double getBeta2() {
        return beta2;
    }

    public void setBeta2(double beta2) {
        this.beta2 = beta2;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
}