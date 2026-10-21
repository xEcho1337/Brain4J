package org.brain4j.core.training.optimizer.impl;

import java.util.HashMap;
import java.util.Map;

import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Applies sign-based updates with momentum interpolation.
 * Use it for memory-efficient training of large models.
 */
public class Lion implements Optimizer {

    public record Config(double beta1, double beta2) {}

    private final Config config;
    private final double learningRate;
    private Map<Tensor, Tensor> momentumHistory;

    /**
     * Create an optimizer with default betas.
     */
    public Lion(double learningRate) {
        this(learningRate, 0.9, 0.99);
    }

    /**
     * Create an optimizer with the given betas.
     */
    public Lion(double learningRate, double beta1, double beta2) {
        this.learningRate = learningRate;
        this.config = new Config(beta1, beta2);

        resetState();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize() {
        resetState();
    }

    private void resetState() {
        this.momentumHistory = new HashMap<>();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double learningRate() {
        return learningRate;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor step(Tensor weights, Tensor gradient, int samples) {
        Tensor normalized = gradient.divide(samples);
        Tensor momentum = momentumFor(weights, normalized);

        Tensor direction = momentum.times(config.beta1())
                .add(normalized.times(1 - config.beta1()).broadcastLike(momentum))
                .sign();

        Tensor newMomentum = updateMomentum(momentum, normalized);

        momentumHistory.put(weights, newMomentum);

        return direction.mul(learningRate);
    }

    private Tensor momentumFor(Tensor weights, Tensor gradient) {
        Tensor momentum = momentumHistory.get(weights);

        if (momentum == null) {
            momentum = Tensors.zerosLike(gradient);

            if (gradient instanceof GpuTensor gpuTensor) {
                momentum = GpuTensor.persistent(momentum, gpuTensor.device());
            }
        }

        return momentum;
    }

    private Tensor updateMomentum(Tensor momentum, Tensor gradient) {
        return momentum.times(config.beta2())
                .add(gradient.times(1 - config.beta2()).broadcastLike(momentum));
    }

    /**
     * Returns the momentum history by weights.
     *
     * @return the history map
     */
    public Map<Tensor, Tensor> momentumHistory() {
        return momentumHistory;
    }

    /**
     * Returns the Lion configuration.
     *
     * @return the configuration
     */
    public Config config() {
        return config;
    }
}
