package org.brain4j.math.operations.matmul;

import org.brain4j.math.tensor.Tensor;

/**
 * Multiplies two tensors into an output tensor. Use it to plug scalar or
 * vectorized kernels into the matmul path.
 */
public interface MatmulProvider {

    /**
     * Multiply inputs and store the result.
     *
     * @param a left operand
     * @param b right operand
     * @param c output receiving the product
     */
    void multiply(Tensor a, Tensor b, Tensor c);
}
