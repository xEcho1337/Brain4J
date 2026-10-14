package org.brain4j.math.activation;

import org.silicon.api.function.ComputeFunction;
import org.silicon.api.kernel.ComputeSize;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.WeightInit;

/**
 * Applies a non-linear function to layer outputs.
 * Use it to let networks learn complex patterns.
 */
public interface Activation {

    /**
     * Returns the weight initialization matching this activation.
     *
     * @return the default weight init
     */
    WeightInit defaultWeightInit();

    /**
     * Applies this activation to a single value.
     *
     * @return the activated value
     */
    double activate(double input);

    /**
     * Returns the derivative at a single value.
     *
     * @return the derivative value
     */
    double derivative(double input);

    /**
     * Returns the GPU kernel prefix for this activation.
     *
     * @return the kernel prefix
     */
    String getKernelPrefix();

    /**
     * Returns the numeric identifier of this activation.
     *
     * @return the activation id
     */
    int getActivationId();

    /**
     * Returns the display name of this activation.
     *
     * @return the activation name
     */
    default String name() {
        return getClass().getSimpleName().replace("Activation", "");
    }

    /**
     * Builds the GPU kernel applying this activation.
     *
     * @return a kernel factory ready to launch
     */
    default KernelFactory createKernel(
        ComputeFunction kernel,
        GpuTensor input,
        GpuTensor output
    ) {
        return KernelFactory
            .create(kernel)
            .intVal(getActivationId())
            .floatVal(0f)
            .intVal(input.size())
            .buffer(input.getDataBuffer())
            .buffer(output.getDataBuffer());
    }

    /**
     * Applies this activation element-wise.
     *
     * @return a new tensor with activated values
     */
    default Tensor activate(Tensor input) {
        int[] shape = input.shape();

        if (input instanceof GpuTensor gpuInput) {
            return computeGpu(gpuInput, "forward");
        }

        float[] inputData = input.data();
        float[] resultData = new float[inputData.length];

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) activate(inputData[i]);
        }

        return Tensors.create(shape, resultData);
    }

    /**
     * Returns the element-wise derivative, scaled by the output gradient when given.
     *
     * @param gradOut downstream gradient, or {@code null} to skip scaling
     * @return the derivative tensor
     */
    default Tensor derivative(Tensor input, Tensor output, Tensor gradOut) {
        int[] shape = input.shape();

        if (input instanceof GpuTensor gpuInput) {
            Tensor derivative = computeGpu(gpuInput, "backward");

            return gradOut == null ? derivative : derivative.mul(gradOut);
        }

        float[] inputData = input.data();
        float[] resultData = new float[inputData.length];

        for (int i = 0; i < resultData.length; i++) {
            resultData[i] = (float) derivative(inputData[i]);
        }

        Tensor derivative = Tensors.create(shape, resultData);

        return gradOut == null ? derivative : gradOut.times(derivative);
    }

    private GpuTensor computeGpu(GpuTensor input, String suffix) {
        Device device = input.getDevice();
        GpuTensor result = new GpuTensor(device, input.shape());

        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            ComputeFunction kernel = GpuContext.findFunction(device, "activation_" + suffix);
            KernelFactory factory = createKernel(kernel, input, result);

            ComputeSize size = new ComputeSize(input.size(), 1, 1);
            factory.launch(queue.queue(), size);
        }

        return result;
    }
}
