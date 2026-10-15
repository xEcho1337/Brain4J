package org.brain4j.math.scaler.impl;

import java.util.List;

import org.brain4j.math.Tensors;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Tensor;

/**
 * Z-score scaler. Use it to standardize features to zero mean and unit variance.
 */
public class ZScoreScaler implements FeatureScaler {

    private float mean;
    private float std;

    /**
     * Creates an unfitted scaler.
     */
    public ZScoreScaler() {
    }

    /**
     * Creates a scaler with known statistics.
     */
    public ZScoreScaler(float mean, float std) {
        this.mean = mean;
        this.std = std;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void fit(List<Tensor> tensors) {
        float sum = 0f;
        int count = 0;

        for (Tensor tensor : tensors) {
            float[] data = tensor.data();

            for (float v : data) {
                sum += v;
                count++;
            }
        }

        this.mean = sum / count;

        double varianceSum = 0f;

        for (Tensor tensor : tensors) {
            float[] data = tensor.data();

            for (float v : data) {
                varianceSum += Math.pow(v - mean, 2);
            }
        }

        this.std = (float) Math.sqrt(varianceSum / count);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor transform(Tensor tensor) {
        float[] data = tensor.data();
        float[] out = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            out[i] = (data[i] - mean) / std;
        }

        return Tensors.create(tensor.shape(), out);
    }

    /**
     * Returns the fitted mean.
     *
     * @return the mean
     */
    public float mean() {
        return mean;
    }

    /**
     * Returns the fitted standard deviation.
     *
     * @return the standard deviation
     */
    public float std() {
        return std;
    }
}
