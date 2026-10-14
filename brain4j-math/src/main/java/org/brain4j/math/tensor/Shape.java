package org.brain4j.math.tensor;

import java.util.Arrays;

import org.brain4j.math.Copyable;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;

/**
 * Holds the dimension sizes of a tensor.
 * Use it to describe shapes when creating or reshaping tensors.
 */
public class Shape implements Copyable<Shape> {

    private final int[] dims;

    protected Shape(int[] dims) {
        this.dims = dims.clone();
    }

    /**
     * Creates a shape from the given dimension sizes.
     *
     * @return a new shape
     */
    public static Shape of(int... dims) {
        return new Shape(dims);
    }

    /**
     * Concatenates several shapes along the last dimension.
     *
     * @return the combined shape
     */
    public static Shape concat(Shape... all) {
        Shape base = all[0];

        for (int i = 1; i < all.length; i++) {
            base = base.concat(all[i], -1);
        }

        return base;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Shape copy() {
        return new Shape(dims);
    }

    /**
     * Returns the concatenation of this shape with another along one dimension.
     *
     * @param dim concatenation dimension, negative counts from the end
     * @return the combined shape
     */
    public Shape concat(Shape other, int dim) {
        if (rank() != other.rank()) {
            throw Commons.illegalArgument("Shapes must have the same rank");
        }

        int rank = rank();
        int dimension = Math.floorMod(dim, rank);

        if (dimension < 0 || dimension >= rank) {
            throw Commons.illegalArgument("Invalid dimension: " + dimension);
        }

        for (int i = 0; i < rank; i++) {
            if (i != dimension && dims[i] != other.dims()[i]) {
                throw Commons.illegalArgument("Shapes must match in all dims except the concatenation one. " +
                    "Current: %s, Other: %s", Arrays.toString(dims), Arrays.toString(other.dims()));
            }
        }

        int[] newShape = Arrays.copyOf(dims, rank);
        newShape[dimension] += other.dims()[dimension];

        return new Shape(newShape);
    }

    /**
     * Returns the size of one dimension, supporting negative indexing.
     *
     * @return the dimension size
     */
    public int dim(int index) {
        if (index < 0) {
            index = Math.floorMod(index, dims.length);
        }

        return dims[index];
    }

    /**
     * Returns the size of the last dimension.
     *
     * @return the last dimension size
     */
    public int last() {
        return dims[dims.length - 1];
    }

    /**
     * Returns the size counted back from the last dimension.
     *
     * @param offset distance from the end (0 is last)
     * @return the dimension size
     */
    public int last(int offset) {
        if (offset > dims.length - 1) {
            throw new IllegalArgumentException("Offset cannot be higher than rank - 1");
        }

        return dims[dims.length - 1 - offset];
    }

    /**
     * Returns the total element count for this shape.
     *
     * @return the number of elements
     */
    public int size() {
        return Tensors.computeSize(dims);
    }

    /**
     * Returns the number of dimensions.
     *
     * @return the rank
     */
    public int rank() {
        return dims.length;
    }

    /**
     * Returns the backing dimension array.
     *
     * @return the dimension sizes
     */
    public int[] dims() {
        return dims;
    }

    /**
     * Returns the sub-shape in {@code [start, end)}.
     *
     * @return the sliced shape
     */
    public Shape slice(int start, int end) {
        end = Math.floorMod(end, dims.length);

        if (end < start) {
            throw new IllegalArgumentException("End must be greater or equal than start!");
        }

        int[] result = new int[end - start];
        System.arraycopy(dims, start, result, 0, result.length);

        return Shape.of(result);
    }

    /**
     * Copies dimensions into the given array at the given offset.
     */
    public void copy(int[] result, int destOffset) {
        if (rank() > result.length) {
            throw Commons.illegalArgument("Rank is higher than result array");
        }

        System.arraycopy(dims, 0, result, destOffset, dims.length);
    }

    /**
     * Copies dimensions into the given array.
     */
    public void copy(int[] result) {
        copy(result, 0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return Arrays.toString(dims);
    }
}
