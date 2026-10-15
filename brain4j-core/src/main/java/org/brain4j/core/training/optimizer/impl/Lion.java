package org.brain4j.core.training.optimizer.impl;

import java.util.HashMap;
import java.util.Map;

import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;

/**
 * Applies sign-based updates with momentum interpolation.
 * Use it for memory-efficient training of large models.
 */
public class Lion extends Optimizer {

    protected Map<Tensor, Tensor> momentumHistory = new HashMap<>();
    protected double beta1 = 0.9;
    protected double beta2 = 0.99;

    protected Lion() {
        super(0);
    }

    /**
     * Create an optimizer with default betas.
     */
    public Lion(double learningRate) {
        super(learningRate);
    }

    /**
     * Create an optimizer with the given betas.
     */
    public Lion(double learningRate, double beta, double beta2) {
        super(learningRate);
        this.beta1 = beta;
        this.beta2 = beta2;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        Tensor scaledGrad = gradient.mul(1 - beta2);

        Tensor newMomentum = calcMomentum(weights, gradient);
        Tensor scaledMomentum = newMomentum.times(beta2);
        Tensor update = scaledMomentum.add(scaledGrad.broadcastLike(scaledMomentum)).sign();

        momentumHistory.put(weights, newMomentum);

        return update;
    }

    /**
     * Compute the interpolated momentum for the given gradient.
     *
     * @return the updated momentum
     */
    public Tensor calcMomentum(Tensor weights, Tensor gradient) {
        Tensor momentum = momentumHistory.getOrDefault(weights, Tensors.zerosLike(gradient));
        Tensor scaledGrad = gradient.mul(1 - beta1);

        return momentum.mul(beta1).add(scaledGrad.broadcastLike(momentum));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize() {
        this.momentumHistory = new HashMap<>();
    }

    /**
     * Return the momentum history by weights.
     *
     * @return the history map
     */
    public Map<Tensor, Tensor> getMomentumHistory() {
        return momentumHistory;
    }

    /**
     * Replace the momentum history.
     *
     * @return this optimizer
     */
    public Lion setMomentumHistory(Map<Tensor, Tensor> momentumHistory) {
        this.momentumHistory = momentumHistory;

        return this;
    }

    /**
     * Return the update interpolation factor.
     *
     * @return the beta1 value
     */
    public double getBeta1() {
        return beta1;
    }

    /**
     * Replace the update interpolation factor.
     *
     * @return this optimizer
     */
    public Lion setBeta1(double beta1) {
        this.beta1 = beta1;

        return this;
    }

    /**
     * Return the momentum tracking factor.
     *
     * @return the beta2 value
     */
    public double getBeta2() {
        return beta2;
    }

    /**
     * Replace the momentum tracking factor.
     *
     * @return this optimizer
     */
    public Lion setBeta2(double beta2) {
        this.beta2 = beta2;

        return this;
    }
}
