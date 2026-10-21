package org.brain4j.math.clipper.impl;

import org.silicon.api.kernel.ComputeSize;

import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Clamps each gradient element to a bound. Use it to stop exploding gradients.
 */
public class HardClipper implements GradientClipper {

    private double bound;

    /**
     * Creates an unbound clipper; set the bound before use.
     */
    public HardClipper() {
    }

    /**
     * Creates a clipper with the given bound.
     *
     * @param bound absolute clamp value, must be non-negative
     */
    public HardClipper(double bound) {
        this.bound = bound;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void clipCpu(CpuTensor grad) {
        grad.map(x -> Math.clamp(x, -bound, bound));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void clipGpu(GpuTensor grad) {
        Device device = grad.device();

        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            ComputeSize size = new ComputeSize(grad.size(), 1, 1);

            KernelFactory.create(device, kernelName())
                .buffer(grad.dataBuffer())
                .floatVal((float) bound)
                .intVal(grad.size())
                .launch(queue.queue(), size);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String kernelName() {
        return "hard_clip";
    }

    /**
     * Returns the clamp bound.
     *
     * @return the absolute bound
     */
    public double bound() {
        return bound;
    }
}
