package org.brain4j.math.clipper.impl;

import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * No-op clipper. Use it to disable gradient clipping.
 */
public class NoClipper implements GradientClipper {

    /**
     * {@inheritDoc}
     */
    @Override
    public void clipCpu(CpuTensor grad) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void clipGpu(GpuTensor grad) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String kernelName() {
        return "";
    }
}
