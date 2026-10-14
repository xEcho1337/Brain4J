package org.brain4j.math.commons;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

/**
 * Holds one batch of inputs and labels.
 * Use it when iterating a dataset during training or evaluation.
 */
public record Batch(Tensor[] first, Tensor[] second) {

    /**
     * Moves all tensors in this batch to the given device.
     *
     * @return a new batch on the target device
     */
    public Batch to(Device device) {
        Tensor[] newInputs = new Tensor[first.length];
        Tensor[] newLabels = new Tensor[second.length];

        for (int i = 0; i < newInputs.length; i++) {
            newInputs[i] = first[i].to(device);
        }

        for (int i = 0; i < newLabels.length; i++) {
            newLabels[i] = second[i].to(device);
        }

        return new Batch(newInputs, newLabels);
    }
}
