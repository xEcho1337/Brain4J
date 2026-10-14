package org.brain4j.math.clipper;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Bounds gradients to keep training stable.
 * Use it to prevent exploding or vanishing gradients.
 */
public interface GradientClipper {

    /**
     * Clips the given gradient in place, dispatching by backend.
     */
    default void clip(Tensor grad) {
        if (grad instanceof CpuTensor gradCpu) {
            clipCpu(gradCpu);
        }

        if (grad instanceof GpuTensor gradGpu) {
            clipGpu(gradGpu);
        }
    }

    /**
     * Clips a CPU gradient tensor in place.
     */
    void clipCpu(CpuTensor grad);

    /**
     * Clips a GPU gradient tensor in place.
     */
    void clipGpu(GpuTensor grad);

    /**
     * Returns the GPU kernel backing this clipper.
     *
     * @return the kernel name
     */
    String kernelName();
}
