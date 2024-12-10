package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;

import java.util.List;

public class Adam extends Optimizer {

    static {
        System.load("/Users/echo/IdeaProjects/Brain4J/src/test/java/nativevec/natives/libbrain4j_backend.dylib");
    }

    // Momentum vectors
    private double[] firstMomentum;
    private double[] secondMomentum;

    private double beta1Timestep;
    private double beta2Timestep;

    private double beta1;
    private double beta2;
    private double epsilon;
    private int timestep;

    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        super(learningRate);
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    public native double[] calculateChange(double firstMomentum, double secondMomentum, double gradient,
                                         double beta1Timestep, double beta2Timestep, double learningRate,
                                         double epsilon, double beta1, double beta2);

    @Override
    public void postInitialize() {
        this.firstMomentum = new double[Synapse.ID_COUNTER];
        this.secondMomentum = new double[Synapse.ID_COUNTER];
    }

    @Override
    public void update(Synapse synapse) {
        double gradient = synapse.getOutputNeuron().getDelta() * synapse.getInputNeuron().getValue();

        int synapseId = synapse.getSynapseId();

        double currentFirstMomentum = firstMomentum[synapseId];
        double currentSecondMomentum = secondMomentum[synapseId];

        double m = beta1 * currentFirstMomentum + (1 - beta1) * gradient;
        double v = beta2 * currentSecondMomentum + (1 - beta2) * gradient * gradient;

        firstMomentum[synapseId] = m;
        secondMomentum[synapseId] = v;

        double mHat = m / (1 - beta1Timestep);
        double vHat = v / (1 - beta2Timestep);

        double deltaWeight = (learningRate * mHat) / (Math.sqrt(vHat) + epsilon);

        /*double[] changes = calculateChange(currentFirstMomentum, currentSecondMomentum, gradient,
                beta1Timestep, beta2Timestep, learningRate, epsilon, beta1, beta2);

        firstMomentum[synapseId] = changes[0];
        secondMomentum[synapseId] = changes[1];

        double deltaWeight = changes[2];*/

        synapse.setWeight(synapse.getWeight() + deltaWeight);
    }

    @Override
    public void postIteration(List<Layer> layers) {
        timestep++;

        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);

        layers.parallelStream().forEach(layer -> {
            for (Synapse synapse : layer.getSynapses()) {
                update(synapse);
            }

            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getDelta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        });
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