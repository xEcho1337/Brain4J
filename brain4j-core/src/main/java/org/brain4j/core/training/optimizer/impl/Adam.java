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
public class Adam extends Optimizer {

    protected Map<Tensor, Tensor> firstMomentum;
    protected Map<Tensor, Tensor> secondMomentum;

    protected double beta1Timestep;
    protected double beta2Timestep;

    protected float beta1;
    protected float beta2;
    protected float epsilon;

    protected int timestep = 1;

    protected Adam() {
        super(0);
    }

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
        super(learningRate);
        this.beta1 = (float) beta1;
        this.beta2 = (float) beta2;
        this.epsilon = (float) epsilon;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initialize() {
        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);
        this.firstMomentum = new HashMap<>();
        this.secondMomentum = new HashMap<>();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        Tensor first = firstMomentum.get(weights);
        Tensor second = secondMomentum.get(weights);

        if (first == null || second == null) {
            first = Tensors.zeros(gradient.shape());
            second = Tensors.zeros(gradient.shape());

            if (gradient instanceof GpuTensor gpuTensor) {
                Device device = gpuTensor.getDevice();
                first = GpuTensor.persistent(first, device);
                second = GpuTensor.persistent(second, device);
            }
        }

        Tensor gradSquared = gradient.times(gradient);

        first.mul(beta1);
        second.mul(beta2);

        Tensor gradBeta1 = gradient.mul(1 - beta1);
        Tensor gradBeta2 = gradSquared.mul(1 - beta2);

        first.add(gradBeta1.broadcastLike(first));
        second.add(gradBeta2.broadcastLike(second));

        firstMomentum.put(weights, first);
        secondMomentum.put(weights, second);

        double biasCorrection1 = 1 - beta1Timestep;
        double biasCorrection2 = 1 - beta2Timestep;

        Tensor mHat = first.divide(biasCorrection1);
        Tensor vHat = second.divide(biasCorrection2);

        return mHat.div(vHat.sqrt().add(epsilon));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void postBatch() {
        this.timestep++;
        this.beta1Timestep *= beta1;
        this.beta2Timestep *= beta2;
    }

    /**
     * Return the first-moment decay.
     *
     * @return the beta1 value
     */
    public double beta1() {
        return beta1;
    }

    /**
     * Replace the first-moment decay.
     */
    public void setBeta1(double beta1) {
        this.beta1 = (float) beta1;
    }

    /**
     * Return the second-moment decay.
     *
     * @return the beta2 value
     */
    public double beta2() {
        return beta2;
    }

    /**
     * Replace the second-moment decay.
     */
    public void setBeta2(double beta2) {
        this.beta2 = (float) beta2;
    }

    /**
     * Return the numerical stability term.
     *
     * @return the epsilon value
     */
    public double epsilon() {
        return epsilon;
    }

    /**
     * Replace the numerical stability term.
     */
    public void setEpsilon(double epsilon) {
        this.epsilon = (float) epsilon;
    }
}
