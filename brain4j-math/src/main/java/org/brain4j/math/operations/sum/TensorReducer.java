package org.brain4j.math.operations.sum;

import org.brain4j.math.tensor.Tensor;

/**
 * Reduces a tensor by summation along one dimension. Use it to implement sum
 * with scalar or vectorized backends.
 */
public interface TensorReducer {
    /**
     * Sum a tensor along a dimension.
     *
     * @param tensor input tensor
     * @param dim dimension to reduce
     * @param keepDim whether to retain the reduced dimension
     * @return reduced tensor
     */
    Tensor sum(Tensor tensor, int dim, boolean keepDim);
}
