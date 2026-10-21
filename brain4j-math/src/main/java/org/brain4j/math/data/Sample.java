package org.brain4j.math.data;

import org.brain4j.math.Copyable;
import org.brain4j.math.tensor.Tensor;

/**
 * Holds one training example as input and label tensors.
 * Use it as the element type of a {@link ListDataSource}.
 */
public record Sample(Tensor[] inputs, Tensor[] labels) implements Copyable<Sample> {

    /**
     * Creates a single-input, single-label sample.
     */
    public Sample(Tensor input, Tensor label) {
        this(new Tensor[] { input }, new Tensor[] { label });
    }

    /**
     * Returns the input tensor at the given position.
     *
     * @return the requested input tensor
     */
    public Tensor input(int index) {
        return inputs[index];
    }

    /**
     * Returns the label tensor at the given position.
     *
     * @return the requested label tensor
     */
    public Tensor label(int index) {
        return labels[index];
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Sample copy() {
        Tensor[] clonedInputs = new Tensor[inputs.length];
        Tensor[] clonedLabels = new Tensor[labels.length];

        for (int i = 0; i < clonedInputs.length; i++) {
            clonedInputs[i] = inputs[i].copy();
        }

        for (int i = 0; i < clonedLabels.length; i++) {
            clonedLabels[i] = labels[i].copy();
        }

        return new Sample(clonedInputs, clonedLabels);
    }
}
