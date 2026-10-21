package org.brain4j.core.training.optimizer.impl;

import java.util.HashMap;
import java.util.Map;

import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Applies adaptive updates with first and second moment estimates.
 * Use it as the default optimizer for fast, stable convergence.
 */
public class Adam implements Optimizer {

    public record Config(double beta1, double beta2, double epsilon) {}

    protected final Adam.Config config;
    protected final Map<Tensor, Tensor> firstMomentum;
    protected final Map<Tensor, Tensor> secondMomentum;

    protected double beta1Timestep;
    protected double beta2Timestep;
    protected double learningRate;
    protected int timestep = 1;

    /**
     * Create an optimizer with default betas and epsilon.
     */
    public Adam(double learningRate) {
        this(learningRate, 0.9, 0.999, 1e-8);
    }

    /**
     * Create an optimizer with the given hyperparameters.
     */
    public Adam(double learningRate, double beta1, double beta2, double epsilon) {
        this.learningRate = learningRate;
        this.config = new Adam.Config(beta1, beta2, epsilon);
        this.firstMomentum = new HashMap<>();
        this.secondMomentum = new HashMap<>();

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
        this.timestep = 1;
        this.beta1Timestep = config.beta1();
        this.beta2Timestep = config.beta2();
        this.firstMomentum.clear();
        this.secondMomentum.clear();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor step(Tensor weights, Tensor gradient, int samples) {
        Tensor normalized = gradient.divide(samples);
        Tensor first = firstMomentum.get(weights);
        Tensor second = secondMomentum.get(weights);

        if (first == null || second == null) {
            first = Tensors.zeros(normalized.shape());
            second = Tensors.zeros(normalized.shape());

            if (normalized instanceof GpuTensor gpuTensor) {
                Device device = gpuTensor.device();
                first = GpuTensor.persistent(first, device);
                second = GpuTensor.persistent(second, device);
            }
        }

        Tensor gradSquared = normalized.times(normalized);

        first.mul(config.beta1());
        second.mul(config.beta2());

        Tensor gradBeta1 = normalized.times(1 - config.beta1());
        Tensor gradBeta2 = gradSquared.times(1 - config.beta2());

        first.add(gradBeta1.broadcastLike(first));
        second.add(gradBeta2.broadcastLike(second));

        firstMomentum.put(weights, first);
        secondMomentum.put(weights, second);

        double biasCorrection1 = 1 - beta1Timestep;
        double biasCorrection2 = 1 - beta2Timestep;

        Tensor mHat = first.divide(biasCorrection1);
        Tensor vHat = second.divide(biasCorrection2);

        return mHat.div(vHat.sqrt().add(config.epsilon())).mul(learningRate);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void postBatch() {
        this.timestep++;
        this.beta1Timestep *= config.beta1();
        this.beta2Timestep *= config.beta2();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double learningRate() {
        return learningRate;
    }

    /**
     * Returns the Adam configuration.
     *
     * @return the configuration
     */
    public Config config() {
        return config;
    }
}
