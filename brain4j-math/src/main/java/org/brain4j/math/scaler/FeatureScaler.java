package org.brain4j.math.scaler;

import java.util.List;

import org.brain4j.math.tensor.Tensor;

/**
 * Normalizes input features before training or inference.
 * Use it to fit scaling statistics once, then transform each tensor.
 */
public interface FeatureScaler {

    /**
     * Computes scaling statistics from the given tensors.
     */
    void fit(List<Tensor> tensors);

    /**
     * Applies the fitted scaling to the given tensor.
     *
     * @return the scaled tensor
     */
    Tensor transform(Tensor tensor);

    /**
     * Fits statistics, then transforms all given tensors.
     *
     * @return the scaled tensors
     */
    default List<Tensor> fitAndTransform(List<Tensor> tensors) {
        fit(tensors);

        return transform(tensors);
    }

    /**
     * Applies the fitted scaling to all given tensors.
     *
     * @return the scaled tensors
     */
    default List<Tensor> transform(List<Tensor> tensors) {
        return tensors.stream().map(this::transform).toList();
    }
}
