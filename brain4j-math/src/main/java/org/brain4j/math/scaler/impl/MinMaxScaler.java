package org.brain4j.math.scaler.impl;

import java.util.List;

import org.brain4j.math.Tensors;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Tensor;

/**
 * Min-max scaler. Use it to map features into a fixed output range.
 */
public class MinMaxScaler implements FeatureScaler {

    private final float rangeMin;
    private final float rangeMax;
    private float dataMin;
    private float dataMax;

    /**
     * Creates a scaler with the given output range.
     */
    public MinMaxScaler(float rangeMin, float rangeMax) {
        this.rangeMin = rangeMin;
        this.rangeMax = rangeMax;
    }

    /**
     * Creates a scaler with known data bounds.
     */
    public MinMaxScaler(float rangeMin, float rangeMax, float dataMin, float dataMax) {
        this.rangeMin = rangeMin;
        this.rangeMax = rangeMax;
        this.dataMin = dataMin;
        this.dataMax = dataMax;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void fit(List<Tensor> tensors) {
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;

        for (Tensor tensor : tensors) {
            float[] data = tensor.data();

            for (float v : data) {
                if (v < min) {
                    min = v;
                }

                if (v > max) {
                    max = v;
                }
            }
        }

        this.dataMin = min;
        this.dataMax = max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor transform(Tensor tensor) {
        float[] data = tensor.data();
        float[] out = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            out[i] = (data[i] - dataMin) / (dataMax - dataMin)
                * (rangeMax - rangeMin) + rangeMin;
        }

        return Tensors.create(tensor.shape(), out);
    }

    /**
     * Returns the output range minimum.
     *
     * @return the range minimum
     */
    public float rangeMin() {
        return rangeMin;
    }

    /**
     * Returns the output range maximum.
     *
     * @return the range maximum
     */
    public float rangeMax() {
        return rangeMax;
    }

    /**
     * Returns the fitted data minimum.
     *
     * @return the data minimum
     */
    public float dataMin() {
        return dataMin;
    }

    /**
     * Returns the fitted data maximum.
     *
     * @return the data maximum
     */
    public float dataMax() {
        return dataMax;
    }
}
