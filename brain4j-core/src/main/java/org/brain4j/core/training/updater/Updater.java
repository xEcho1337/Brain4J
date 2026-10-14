package org.brain4j.core.training.updater;

import java.util.HashMap;
import java.util.Map;

import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Accumulates gradients and applies them to weights.
 * Use it to define when batched gradients update the model.
 *
 * @see StochasticUpdater
 * @see NormalUpdater
 */
public abstract class Updater {

    protected final Map<Tensor, Tensor> weightsGradients = new HashMap<>();

    /**
     * Persistent GPU accumulation buffers reused across batches.
     */
    protected final Map<Tensor, Tensor> persistentStorage = new HashMap<>();

    /**
     * Applies accumulated gradients to weights in place.
     *
     * @param learningRate the step size
     * @param samples the sample count used to normalize gradients
     */
    public void updateWeights(double learningRate, int samples) {
        for (Map.Entry<Tensor, Tensor> entry : weightsGradients.entrySet()) {
            Tensor weights = entry.getKey();
            Tensor gradient = entry.getValue();

            if (gradient == null || weights == null) {
                continue;
            }

            while (gradient.rank() > weights.rank()) {
                gradient = gradient.sum(0, false);
            }

            weights.sub(gradient.div(samples).mul(learningRate));
        }
    }

    /**
     * Accumulates a gradient for the given weights.
     */
    public void change(Tensor weights, Tensor gradient) {
        if (!(gradient instanceof GpuTensor gpuGradient)) {
            weightsGradients.merge(weights, gradient, Tensor::add);
            return;
        }

        Tensor stored = weightsGradients.get(weights);

        if (stored == null) {
            stored = persistentStorage.get(weights);

            if (stored == null) {
                stored = GpuTensor.persistent(Tensors.zeros(gradient.shape()), gpuGradient.getDevice());
                persistentStorage.put(weights, stored);
            } else {
                // Reuse the buffer instead of leaking GPU memory.
                stored.mul(0);
            }
        }

        stored.add(gradient);
        weightsGradients.put(weights, stored);
    }

    /**
     * Clears all accumulated gradients.
     */
    public void initialize() {
        weightsGradients.clear();
    }

    /**
     * Runs a hook after training over the dataset.
     *
     * @param learningRate the step size
     * @param samples the training sample count
     */
    public void postFit(double learningRate, int samples) {
    }

    /**
     * Runs a hook after each mini-batch.
     *
     * @param learningRate the step size
     * @param samples the batch sample count
     */
    public void postBatch(double learningRate, int samples) {
    }
}
