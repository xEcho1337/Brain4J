package org.brain4j.math.operations.matmul;

/**
 * Carries buffers and shapes for a batched matrix multiplication. Use it to
 * share one immutable multiplication description across parallel tasks.
 *
 * @param A left operand buffer
 * @param B right operand buffer
 * @param C output buffer
 * @param m rows of A and C
 * @param n columns of A and rows of B
 * @param p columns of B and C
 * @param transposedA whether A is read transposed
 * @param transposedB whether B is read transposed
 * @param batchA batch count of A
 * @param batchB batch count of B
 */
public record MatmulParameters(
    float[] A, float[] B, float[] C,
    int m, int n, int p,
    boolean transposedA,
    boolean transposedB,
    int batchA, int batchB
) {
    /**
     * Get the row-column product of the left operand.
     *
     * @return m times n
     */
    public int mn() {
        return m * n;
    }

    /**
     * Get the inner-output product of the right operand.
     *
     * @return n times p
     */
    public int np() {
        return n * p;
    }

    /**
     * Get the output matrix size.
     *
     * @return m times p
     */
    public int mp() {
        return m * p;
    }
}
