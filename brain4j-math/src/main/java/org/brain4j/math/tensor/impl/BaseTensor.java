package org.brain4j.math.tensor.impl;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.commons.D2DFunction;
import org.brain4j.math.commons.Range;
import org.brain4j.math.convolution.pooling.impl.MaxPooling;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.operations.ParallelMap;
import org.brain4j.math.operations.sum.TensorReducer;
import org.brain4j.math.operations.sum.impl.SIMDTensorReducer;
import org.brain4j.math.operations.sum.impl.ScalarTensorReducer;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.AutogradContext;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.impl.ActivationOperation;
import org.brain4j.math.tensor.autograd.impl.AddOperation;
import org.brain4j.math.tensor.autograd.impl.ConcatOperation;
import org.brain4j.math.tensor.autograd.impl.ConvolveOperation;
import org.brain4j.math.tensor.autograd.impl.DivOperation;
import org.brain4j.math.tensor.autograd.impl.GatherOperation;
import org.brain4j.math.tensor.autograd.impl.LayerNormOperation;
import org.brain4j.math.tensor.autograd.impl.MatMulOperation;
import org.brain4j.math.tensor.autograd.impl.MaxPoolOperation;
import org.brain4j.math.tensor.autograd.impl.MulOperation;
import org.brain4j.math.tensor.autograd.impl.RMSNormOperation;
import org.brain4j.math.tensor.autograd.impl.ReshapeOperation;
import org.brain4j.math.tensor.autograd.impl.SliceOperation;
import org.brain4j.math.tensor.autograd.impl.SqueezeOperation;
import org.brain4j.math.tensor.autograd.impl.SubOperation;
import org.brain4j.math.tensor.autograd.impl.TransposeOperation;

import static org.brain4j.math.Tensors.ones;

/**
 * Shares shape, stride, and autograd plumbing for tensors. Use it as the base
 * for CPU and GPU implementations.
 */
public abstract class BaseTensor implements Tensor {

    protected AutogradContext autogradContext;
    protected int[] shape;
    protected int[] strides;
    protected float[] data;
    protected boolean transposed;

    /**
     * Append a formatted tensor view to a builder.
     *
     * @param result builder receiving the text
     * @param dim current dimension
     * @param indices current indices
     * @param format float format
     */
    protected void appendTensor(StringBuilder result, int dim, int[] indices, String format) {
        if (dim == shape.length - 1) {
            result.append("[");

            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;

                float value = get(indices);
                result.append(format.formatted(value));

                if (i < shape[dim] - 1) {
                    result.append(", ");
                }
            }

            result.append("]");
        } else {

            result.append("[");

            for (int i = 0; i < shape[dim]; i++) {
                indices[dim] = i;
                appendTensor(result, dim + 1, indices, format);

                if (i < shape[dim] - 1) {
                    result.append(",\n");
                    result.repeat(" ", dim + 1);
                }
            }

            result.append("]");
        }
    }

    /**
     * Copy a sliced region into a result tensor.
     *
     * @param result destination tensor
     * @param ranges one range per dimension
     * @param srcIndices current source indices
     * @param dstIndices current destination indices
     * @param dim current dimension
     */
    protected void sliceCopy(
        Tensor result,
        Range[] ranges,
        int[] srcIndices,
        int[] dstIndices,
        int dim
    ) {
        if (dim == shape.length) {
            result.set(get(srcIndices), dstIndices);
            return;
        }

        Range range = dim < ranges.length ? ranges[dim] : null;
        int dimension = shape[dim];

        int start = range == null ? 0 : range.start(dimension);
        int end = range == null ? shape[dim] : range.end(dimension);
        int step = range == null ? 1 : range.step();

        if (dim == shape.length - 1 && step == 1) {
            int length = end - start;

            srcIndices[dim] = start;
            dstIndices[dim] = 0;

            int srcOffset = Tensors.flattenIndex(srcIndices, this.strides);
            int dstOffset = Tensors.flattenIndex(dstIndices, result.strides());

            System.arraycopy(
                this.data, srcOffset,
                result.data(), dstOffset,
                length
            );
            return;
        }

        for (int i = start, j = 0; i < end; i += step, j++) {
            srcIndices[dim] = i;
            dstIndices[dim] = j;
            sliceCopy(result, ranges, srcIndices, dstIndices, dim + 1);
        }
    }
    
    /**
     * Apply softmax in place over one segment.
     *
     * @param data buffer to normalize
     * @param offset segment start
     * @param length segment length
     * @param temperature softmax temperature, must be positive
     */
    protected void softmax1D(float[] data, int offset, int length, double temperature) {
        float max = Float.NEGATIVE_INFINITY;
        
        for (int i = 0; i < length; i++) {
            float value = data[offset + i];

            if (value > max) {
                max = value;
            }
        }

        float sum = 0f;
        
        for (int i = 0; i < length; i++) {
            float e = (float) Math.exp((data[offset + i] - max) / temperature);
            sum += e;
            data[offset + i] = e;
        }
        
        for (int i = 0; i < length; i++) {
            data[offset + i] /= sum;
        }
    }
    
    /**
     * Compute softmax over a flat tensor.
     *
     * @param temperature softmax temperature, must be positive
     * @return normalized copy
     */
    protected Tensor softmax1D(double temperature) {
        Tensor result = copy();
        softmax1D(result.data(), 0, result.elements(), temperature);
        return result;
    }

    /**
     * Compute softmax over rows of a 2D tensor.
     *
     * @param temperature softmax temperature, must be positive
     * @return normalized copy
     */
    protected Tensor softmax2D(double temperature) {
        Tensor result = copy();
        
        int rows = shape[0];
        int cols = shape[1];
        
        for (int r = 0; r < rows; r++) {
            softmax1D(result.data(), r * cols, cols, temperature);
        }
        
        return result;
    }
    
    /**
     * Compute softmax over the last axis of a 3D tensor.
     *
     * @param temperature softmax temperature, must be positive
     * @return normalized copy
     */
    protected Tensor softmax3D(double temperature) {
        Tensor result = copy();
        
        int batches = shape[0];
        int rows = shape[1];
        int cols = shape[2];
        
        float[] data = result.data();
        int strideBR = rows * cols;
        
        IntStream.range(0, batches * rows).parallel().forEach(i -> {
            int b = i / rows;
            int r = i % rows;
            softmax1D(data, b * strideBR + r * cols, cols, temperature);
        });
        
        return result;
    }
    
    /**
     * Compute softmax over the last axis of an N-dimensional tensor.
     *
     * @param temperature softmax temperature, must be positive
     * @return normalized copy
     */
    protected Tensor softmaxND(double temperature) {
        Tensor result = copy();
        
        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int outerSize = 1;
        
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }
        
        float[] data = result.data();

        IntStream.range(0, outerSize)
            .parallel()
            .forEach(outer -> softmax1D(data, outer * lastDim, lastDim, temperature));

        return result;
    }
    
    /**
     * Write a slice recursively into this tensor.
     *
     * @param index current indices
     * @param dim current dimension
     * @param offset slice offset
     * @param sliceSize slice length
     * @param input source tensor
     */
    protected void setSliceRecursive(int[] index, int dim, int offset, int sliceSize, Tensor input) {
        if (dim == index.length - 1) {

            for (int i = 0; i < sliceSize; i++) {
                index[dim] = offset + i;

                int[] inputIndex = Arrays.copyOf(index, index.length);
                inputIndex[dim] = i;

                float value = input.get(inputIndex);
                this.set(value, index);
            }

            return;
        }

        for (int i = 0; i < this.shape()[dim]; i++) {
            index[dim] = i;
            setSliceRecursive(index, dim + 1, offset, sliceSize, input);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int shapeAt(int index) {
        return shape[Math.floorMod(index, shape.length)];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int[] shape() {
        return shape;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float[] data() {
        return data;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int[] strides() {
        return strides;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public byte[] toByteArray() {
        float[] data = data();
        
        ByteBuffer buffer = ByteBuffer.allocate(data.length * 4).order(ByteOrder.nativeOrder());

        for (float value : data) {
            buffer.putFloat(value);
        }

        return buffer.array();
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public int linearIndex(int... indices) {
        if (indices.length != shape.length) {
            throw Commons.illegalArgument("Number of indices does not match tensor rank!");
        }

        for (int i = 0; i < indices.length; i++) {
            int index = indices[i];
            
            if (index < 0 || index >= shape[i]) {
                throw Commons.indexOOB("Index %s for dimension %s is out of bounds [0, %s)", index, i, shape[i]);
            }
        }

        return Tensors.flattenIndex(indices, strides);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public float get(int... indices) {
        return data()[linearIndex(indices)];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor set(float value, int... indices) {
        data[linearIndex(indices)] = value;
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int rank() {
        return shape.length;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int elements() {
        return data().length;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int argmax() {
        float[] data = data();

        double max = Double.NEGATIVE_INFINITY;
        int maxIndex = 0;

        for (int i = 0; i < data.length; i++) {

            if (data[i] > max) {
                max = data[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor flatten() {
        return reshape(elements());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor convolve(Tensor kernel) {
        return convolve(kernel, 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor convolve(Tensor kernel, int stride) {
        return Tensors.convolve(this, kernel, stride);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor layerNorm(double epsilon) {
        switch (rank()) {
            case 1 -> layerNorm1D(0, epsilon);
            case 2 -> layerNorm2D(epsilon);
            case 3 -> layerNorm3D(epsilon);
            default -> layerNormND(epsilon);
        }
        
        return this;
    }
    
    private void layerNorm1D(int offset, double epsilon) {
        int features = shape[rank() - 1];
        float mean = 0f;
        
        for (int j = 0; j < features; j++) {
            mean += data[offset + j];
        }
        
        mean /= features;
        
        float var = 0f;
        
        for (int j = 0; j < features; j++) {
            float x = data[offset + j];
            float diff = x - mean;
            
            var += diff * diff;
        }
        
        var /= features;
        
        float invDenom = (float) (1.0f / Math.sqrt(var + epsilon));
        
        for (int j = 0; j < features; j++) {
            int idx = offset + j;
            data[idx] = (data[idx] - mean) * invDenom;
        }
    }
    
    private void layerNorm2D(double epsilon) {
        int rows = shape[0];
        int cols = shape[1];
        
        for (int r = 0; r < rows; r++) {
            layerNorm1D(r * cols, epsilon);
        }
    }
    
    private void layerNorm3D(double epsilon) {
        int batches = shape[0];
        int rows = shape[1];
        int cols = shape[2];
        
        int strideBR = rows * cols;
        
        IntStream.range(0, batches * rows).parallel().forEach(i -> {
            int b = i / rows;
            int r = i % rows;
            layerNorm1D(b * strideBR + r * cols, epsilon);
        });
    }
    
    private void layerNormND(double epsilon) {
        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int outerSize = 1;
        
        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }
        
        IntStream.range(0, outerSize)
            .parallel()
            .forEach(outer -> layerNorm1D(outer * lastDim, epsilon));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double distance(Tensor other) {
        return Math.sqrt(distanceSquared(other));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double distanceSquared(Tensor other) {
        double sum = 0;
        float[] cached = data();

        for (int i = 0; i < cached.length; i++) {
            double diff = cached[i] - other.data()[i];
            sum += diff * diff;
        }

        return sum;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor squeeze() {
        int count = 0;
        
        for (int dim : shape) {

            if (dim != 1) {
                count++;
            }

        }
        
        if (count == rank()) {
            return this;
        }
        
        int[] newShape = new int[count];
        int idx = 0;
        
        for (int dim : shape) {

            if (dim != 1) {
                newShape[idx++] = dim;
            }
        }
        
        return reshape(newShape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor squeeze(int dim) {
        dim = Commons.mod(dim, shape.length);

        if (dim >= rank()) {
            throw new IllegalArgumentException("Dimension must be less than the rank!");
        }

        if (shape[dim] != 1) {
            return this;
        }

        int[] newShape = new int[shape.length - 1];
        int idx = 0;

        for (int i = 0; i < shape.length; i++) {

            if (i != dim) {
                newShape[idx++] = shape[i];
            }
        }
        
        return reshape(newShape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor unsqueeze() {
        return unsqueeze(0);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor broadcast(int[] targetShape) {
        if (Arrays.equals(shape, targetShape)) {
            return this;
        }
        
        int targetRank = targetShape.length;
        int srcRank = shape.length;

        int[] alignedSrcShape = new int[targetRank];
        int pad = targetRank - srcRank;

        for (int i = 0; i < pad; i++) {
            alignedSrcShape[i] = 1;
        }

        System.arraycopy(shape, 0, alignedSrcShape, pad, srcRank);

        for (int d = 0; d < targetRank; d++) {

            if (alignedSrcShape[d] > targetShape[d]) {
                throw new IllegalArgumentException("Cannot broadcast: source dimension " +
                    alignedSrcShape[d] + " > target dimension " + targetShape[d] + " at axis " + d);
            }
        }

        Tensor out = Tensors.zeros(targetShape);

        int total = out.elements();
        int[] srcCoords = new int[targetRank];

        for (int i = 0; i < total; i++) {
            int[] coords = Tensors.unravelIndex(i, targetShape);
            boolean useZero = false;

            for (int d = 0; d < targetRank; d++) {
                int s = alignedSrcShape[d];

                if (s == targetShape[d]) {
                    srcCoords[d] = coords[d];
                } else if (s == 1) {

                    srcCoords[d] = 0;
                } else {

                    if (coords[d] < s) {
                        srcCoords[d] = coords[d];
                    } else {

                        useZero = true;
                        break;
                    }
                }
            }

            if (!useZero) {
                out.set(get(srcCoords), coords);
            }
        }

        return out;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor broadcastLike(Tensor other) {
        return broadcast(other.shape());
    }
    
    public Tensor unsqueeze(int dim) {
        if (dim < 0) {
            dim = Commons.mod(dim, shape.length);
        }
        
        if (dim > shape.length) {
            throw new IllegalArgumentException("Invalid dimension for unsqueeze: " + dim);
        }

        int[] newShape = new int[shape.length + 1];

        for (int i = 0, j = 0; i < newShape.length; i++) {

            if (i == dim) {
                newShape[i] = 1;
            } else {

                newShape[i] = shape[j++];
            }
        }
        
        Tensor result = reshape(newShape);
        result.setAutogradContext(autogradContext);
        
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor transpose() {
        int rank = shape.length;
        return transpose(rank - 2, rank - 1);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor transpose(int dim1, int dim2) {
        int rank = shape.length;
        
        if (rank == 1) {
            return reshape(1, elements());
        }
        
        int[] newShape = shape.clone();
        newShape[dim1] = shape[dim2];
        newShape[dim2] = shape[dim1];
        
        int[] newStrides = strides.clone();
        newStrides[dim2] = strides[dim1];
        newStrides[dim1] = strides[dim2];
        
        BaseTensor view = (BaseTensor) Tensors.create(newShape, newStrides, data);
        view.transposed = !transposed;
        
        return view;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public boolean transposed() {
        return transposed;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double sum() {
        double sum = 0.0;
        float[] data = data();

        for (float value : data) {
            sum += value;
        }

        return sum;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double mean() {
        return sum() / elements();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double variance() {
        double mean = mean();
        double variance = 0.0;

        for (float value : data) {
            variance += Math.pow(value - mean, 2);
        }

        return variance / data.length;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double max() {
        double max = Double.NEGATIVE_INFINITY;

        for (float value : data) {
            max = Math.max(max, value);
        }

        return max;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double min() {
        double min = Double.POSITIVE_INFINITY;

        for (float value : data) {
            min = Math.min(min, value);
        }

        return min;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor sum(int dim, boolean keepDim) {
        dim = Commons.mod(dim, shape.length);

        TensorReducer reducer = DeviceUtils.isSimdAvailable() ? new SIMDTensorReducer() : new ScalarTensorReducer();
        Tensor result = reducer.sum(this, dim, keepDim);
        
        result.setAutogradContext(autogradContext);
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor mean(int dim, boolean keepDim) {
        dim = Commons.mod(dim, shape.length);
        float divisor = shape[dim];

        Tensor summed = this.sum(dim, keepDim);
        Tensor result = summed.div(divisor);

        result.setAutogradContext(autogradContext);
        return result;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor variance(int dim, boolean keepDim) {
        return variance(mean(dim, keepDim), dim, keepDim);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor variance(Tensor mean, int dim, boolean keepDim) {
        dim = Commons.mod(dim, shape.length);

        Tensor meanFirstSquare = copy().pow(2).mean(dim, keepDim);
        Tensor meanSecondSquare = mean.copy().pow(2);
        
        return meanFirstSquare.sub(meanSecondSquare);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor sign() {
        Tensor result = Tensors.zeros(shape);

        for (int i = 0; i < data.length; i++) {
            result.data()[i] = Math.signum(data[i]);
        }

        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor reshape(int... newShape) {
        int newSize = Tensors.computeSize(newShape);

        if (newSize != data().length) {
            throw new IllegalArgumentException(
                    "The total new dimension (" + newSize + ") does not match the current dimension (" + data().length + ")"
            );
        }

        return Tensors.create(newShape, data());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor concat(Tensor other) {
        if (shape.length != other.shape().length) {
            throw new IllegalArgumentException("Concatenation is only supported for tensors with the same number of dimensions.");
        }

        for (int i = 0; i < shape.length - 1; i++) {

            if (shape[i] != other.shape()[i]) {
                throw new IllegalArgumentException("Shapes must match on all dimensions except the last.");
            }
        }

        int rank = shape.length;
        int lastDim = shape[rank - 1];
        int otherLastDim = other.shape()[rank - 1];
        int concatLastDim = lastDim + otherLastDim;

        int[] newShape = Arrays.copyOf(shape, rank);
        newShape[rank - 1] = concatLastDim;

        int outerSize = 1;

        for (int i = 0; i < rank - 1; i++) {
            outerSize *= shape[i];
        }

        float[] resultData = new float[outerSize * concatLastDim];
        float[] thisData = this.data();
        float[] otherData = other.data();

        for (int i = 0; i < outerSize; i++) {
            System.arraycopy(thisData, i * lastDim, resultData, i * concatLastDim, lastDim);
            System.arraycopy(otherData, i * otherLastDim, resultData, i * concatLastDim + lastDim, otherLastDim);
        }

        return Tensors.create(newShape, resultData);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor concat(Tensor other, int dimension) {
        if (shape.length != other.shape().length) {
            throw new IllegalArgumentException("Tensors must have the same rank.");
        }

        int rank = rank();

        dimension = Math.floorMod(dimension, rank);

        if (dimension < 0 || dimension >= rank) {
            throw new IllegalArgumentException("Invalid dimension: " + dimension);
        }

        for (int i = 0; i < rank; i++) {

            if (i != dimension && shape[i] != other.shape()[i]) {
                throw Commons.illegalArgument("Shapes must match in all dims except the concatenation one. " +
                    "Current: %s, Other: %s", Arrays.toString(shape), Arrays.toString(other.shape()));
            }
        }

        int[] newShape = Arrays.copyOf(shape, rank);
        newShape[dimension] += other.shape()[dimension];

        int blockSize = 1;
        int numBlocks = 1;

        for (int i = dimension + 1; i < rank; i++) {
            blockSize *= shape[i];
        }

        for (int i = 0; i < dimension; i++) {
            numBlocks *= shape[i];
        }

        int thisDim = shape[dimension];
        int otherDim = other.shape()[dimension];
        float[] result = new float[numBlocks * (thisDim + otherDim) * blockSize];

        float[] a = this.data();
        float[] b = other.data();

        int resultOffset = 0;
        int thisOffset = 0;
        int otherOffset = 0;

        for (int i = 0; i < numBlocks; i++) {
            int thisBlock = thisDim * blockSize;
            int otherBlock = otherDim * blockSize;

            System.arraycopy(a, thisOffset, result, resultOffset, thisBlock);
            thisOffset += thisBlock;
            resultOffset += thisBlock;

            System.arraycopy(b, otherOffset, result, resultOffset, otherBlock);
            otherOffset += otherBlock;
            resultOffset += otherBlock;
        }

        return Tensors.create(newShape, result);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor activate(Activation activation) {
        return activation.activate(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor select(int dim, int index) {
        dim = Commons.mod(dim, shape.length);

        if (dim >= rank()) {
            throw Commons.illegalArgument("Dimension must be less than the rank!");
        }

        int size = shape[dim];
        int normalizedIndex = index < 0 ? index + size : index;

        if (normalizedIndex < 0 || normalizedIndex >= size) {
            throw Commons.indexOOB("Index %s for dimension %s is out of bounds [0, %s)", normalizedIndex, dim, size);
        }

        Range[] ranges = new Range[shape.length];
        ranges[dim] = Range.point(normalizedIndex);

        return slice(ranges).squeeze(dim);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor slice(Range... ranges) {
        if (ranges.length > shape.length) {
            throw new IllegalArgumentException("Too many ranges specified");
        }

        int[] newShape = new int[shape.length];

        for (int i = 0; i < shape.length; i++) {
            int dim = shape[i];
            Range range = i >= ranges.length ? null : ranges[i];
            newShape[i] = range != null ? range.size(dim) : dim;
        }

        Tensor result = new CpuTensor(Shape.of(newShape));

        int[] srcIndices = new int[shape.length];
        int[] dstIndices = new int[shape.length];

        sliceCopy(result, ranges, srcIndices, dstIndices, 0);

        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor map(D2DFunction function) {
        ParallelMap.map(function, data);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor fill(float value) {
        Arrays.fill(data, value);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor fill(Supplier<Double> supplier) {
        for (int i = 0; i < data.length; i++) {
            data[i] = supplier.get().floatValue();
        }

        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public AutogradContext autogradContext() {
        return autogradContext;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setAutogradContext(AutogradContext context) {
        this.autogradContext = context;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor withGrad() {
        this.autogradContext = new AutogradContext(this, true);
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor noGrad() {
        this.autogradContext = null;
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public boolean usesGrad() {
        return autogradContext != null && autogradContext.requiresGrad();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void zeroGrad() {
        if (autogradContext != null) {
            autogradContext.zeroGrad();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor grad() {
        if (autogradContext != null) {
            return autogradContext.grad();
        }

        return null;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void backward() {
        backward(ones(shape));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void backward(Tensor gradOutput) {
        Objects.requireNonNull(autogradContext, "Autograd is not enabled for this tensor");
        autogradContext.backward(gradOutput);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor forward(Operation operation) {
        if (operation.requiredInputs() != 1) {
            throw new IllegalArgumentException(
                "This operation requires " + operation.requiredInputs() + " inputs! Received 1 instead."
            );
        }

        Tensor result = operation.compute(this);

        if (result.autogradContext() == null) {
            result.setAutogradContext(new AutogradContext(result, true));
        }

        result.autogradContext().setOperation(operation, this);
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor forward(Operation operation, Tensor other) {
        if (operation.requiredInputs() != 2) {
            throw new IllegalArgumentException(
                "This operation requires " + operation.requiredInputs() + " inputs! Received 2 instead."
            );
        }

        Tensor result = operation.compute(this, other);

        if (result.autogradContext() == null) {
            result.setAutogradContext(new AutogradContext(result, true));
        }

        result.autogradContext().setOperation(operation, this, other);
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor forward(Operation operation, Tensor... others) {
        List<Tensor> allInputs = new ArrayList<>();

        allInputs.add(this);
        allInputs.addAll(Arrays.asList(others));

        if (allInputs.size() != operation.requiredInputs()) {
            throw Commons.illegalArgument("This operation requires %d inputs, but %d were received.",
                operation.requiredInputs(), allInputs.size());
        }

        Tensor[] allInputsArray = allInputs.toArray(new Tensor[0]);
        Tensor result = operation.compute(allInputsArray);

        if (result.autogradContext() == null) {
            result.setAutogradContext(new AutogradContext(result, true));
        }

        result.autogradContext().setOperation(operation, allInputsArray);
        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor addGrad(Tensor other) {
        if (!usesGrad()) {
            return plus(other);
        }

        return forward(new AddOperation(), other);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor subGrad(Tensor other) {
        if (!usesGrad()) {
            return minus(other);
        }
        
        return forward(new SubOperation(), other);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor mulGrad(Tensor other) {
        if (!usesGrad()) {
            return times(other);
        }

        return forward(new MulOperation(), other);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor divGrad(Tensor other) {
        if (!usesGrad()) {
            return divide(other);
        }

        return forward(new DivOperation(), other);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor sliceGrad(Range... ranges) {
        if (!usesGrad()) {
            return slice(ranges);
        }
        
        return forward(new SliceOperation(ranges));
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor matmulGrad(Tensor other) {
        if (!usesGrad()) {
            return matmul(other);
        }

        return forward(new MatMulOperation(), other);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor convolveGrad(Tensor other) {
        return convolveGrad(other, 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor convolveGrad(Tensor other, int stride) {
        if (!usesGrad()) {
            return convolve(other, stride);
        }

        return forward(new ConvolveOperation(stride), other);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor layerNormGrad(Tensor weight, Tensor bias, double epsilon) {
        if (!usesGrad() && !weight.usesGrad() && !bias.usesGrad()) {
            return copy().layerNorm(epsilon).mul(weight).add(bias);
        }

        return forward(new LayerNormOperation(epsilon), weight, bias);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor rmsNormGrad(Tensor weight, double epsilon) {
        if (!usesGrad() && !weight.usesGrad()) {
            Tensor owned = copy();
            int features = owned.shape()[owned.rank() - 1];
            Tensor rms = owned.times(owned).sum(-1, true).divide(features).add(epsilon).sqrt();
            return owned.divide(rms).mul(weight);
        }

        return forward(new RMSNormOperation(epsilon), weight);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor gatherGrad(Tensor table) {
        return forward(new GatherOperation(), table);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor maxPoolGrad(int stride, int windowHeight, int windowWidth) {
        if (!usesGrad()) {
            MaxPooling pooling = new MaxPooling(stride, windowHeight, windowWidth);
            return pooling.pool(this);
        }

        return forward(new MaxPoolOperation(stride, windowHeight, windowWidth));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor transposeGrad() {
        if (!usesGrad()) {
            return transpose();
        }

        int rank = rank();
        return forward(new TransposeOperation(rank - 2, rank - 1));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor transposeGrad(int dim1, int dim2) {
        if (!usesGrad()) {
            return transpose(dim1, dim2);
        }

        return forward(new TransposeOperation(dim1, dim2));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor activateGrad(Activation activation) {
        if (!usesGrad()) {
            return activate(activation);
        }

        return forward(new ActivationOperation(activation));
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor concatGrad(Tensor other, int dim) {
        if (!usesGrad()) {
            return concat(other, dim);
        }

        return forward(new ConcatOperation(dim), other);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor reshapeGrad(int... newShape) {
        if (!usesGrad()) {
            return reshape(newShape);
        }

        return forward(new ReshapeOperation(newShape));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor squeezeGrad() {
        if (!usesGrad()) {
            return squeeze();
        }

        return forward(new SqueezeOperation());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor squeezeGrad(int dimension) {
        if (!usesGrad()) {
            return squeeze(dimension);
        }

        return forward(new SqueezeOperation(dimension));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor flip() {
        Tensor result = Tensors.zeros(shape);
        int dims = shape.length;

        int total = elements();

        for (int linear = 0; linear < total; linear++) {
            int[] indices = Tensors.unravelIndex(linear, shape);
            int[] flipped = indices.clone();

            if (dims >= 2) {
                flipped[dims - 1] = shape[dims - 1] - 1 - indices[dims - 1];
                flipped[dims - 2] = shape[dims - 2] - 1 - indices[dims - 2];
            }

            float value = this.get(indices);
            result.set(value, flipped);
        }

        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor softmax() {
        return softmax(1.0);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor softmax(double temperature) {
        return switch (rank()) {
            case 1 -> softmax1D(temperature);
            case 2 -> softmax2D(temperature);
            case 3 -> softmax3D(temperature);
            default -> softmaxND(temperature);
        };
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public String toString(String format) {
        if (shape.length == 0) {
            return format.formatted(data[0]);
        }

        StringBuilder result = new StringBuilder();
        appendTensor(result, 0, new int[shape.length], format);

        return result.toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return toString("%.3f");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Iterator<Float> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            /**
             * {@inheritDoc}
             */
            @Override
            public boolean hasNext() {
                return currentIndex < data.length;
            }

            /**
             * {@inheritDoc}
             */
            @Override
            public Float next() {
                return data[currentIndex++];
            }

        };
    }
}
