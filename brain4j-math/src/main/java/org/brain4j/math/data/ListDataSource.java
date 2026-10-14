package org.brain4j.math.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Splits samples into fixed-size batches for training or evaluation.
 * Use it to iterate batches, optionally shuffled or normalized.
 */
public class ListDataSource implements Cloneable, Iterable<Sample> {

    protected final List<Tensor[]> batchedInputs;
    protected final List<Tensor[]> batchedLabels;
    protected final int batches;

    protected List<Sample> samples;
    protected Device device;
    protected int cursor;
    protected int batchSize;

    /**
     * Creates a CPU-backed source, optionally shuffling before batching.
     */
    public ListDataSource(List<Sample> samples, boolean shuffle, int batchSize) {
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.batches = (samples.size() + batchSize - 1) / batchSize;
        this.batchSize = batchSize;

        if (shuffle) {
            Collections.shuffle(this.samples);
        }

        computeBatches();
    }

    /**
     * Creates a source staging batches on the given device.
     */
    public ListDataSource(Device device, List<Sample> samples, boolean shuffle, int batchSize) {
        this.device = device;
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.batches = (samples.size() + batchSize - 1) / batchSize;
        this.batchSize = batchSize;

        if (shuffle) {
            Collections.shuffle(this.samples);
        }

        computeBatches();
    }

    private ListDataSource(List<Sample> samples, Device device, int batches, int cursor, int batchSize) {
        this.device = device;
        this.samples = samples;
        this.batchedInputs = new ArrayList<>();
        this.batchedLabels = new ArrayList<>();
        this.cursor = cursor;
        this.batches = batches;
        this.batchSize = batchSize;

        computeBatches();
    }

    /**
     * Checks whether more batches remain.
     *
     * @return true if another batch is available
     */
    public boolean hasNext() {
        return cursor < batches;
    }

    /**
     * Resets iteration to the first batch.
     */
    public void reset() {
        cursor = 0;
    }

    /**
     * Applies z-score normalization in place, then recomputes batches.
     *
     * @return this source after normalization
     */
    public ListDataSource normalize() {
        if (samples.isEmpty()) {
            return this;
        }

        int numInputs = samples.getFirst().inputs().length;
        List<List<Tensor>> inputStreams = new ArrayList<>(numInputs);

        for (int i = 0; i < numInputs; i++) {
            inputStreams.add(new ArrayList<>());
        }

        for (Sample sample : samples) {
            Tensor[] inputs = sample.inputs();

            for (int i = 0; i < inputs.length; i++) {
                inputStreams.get(i).add(inputs[i]);
            }
        }

        for (int i = 0; i < numInputs; i++) {
            List<Tensor> tensors = inputStreams.get(i);
            int features = tensors.getFirst().elements();

            float[] means = new float[features];
            float[] stds = new float[features];

            for (int f = 0; f < features; f++) {
                double mean = 0;
                double std = 0;

                for (Tensor tensor : tensors) {
                    float value = tensor.get(f);
                    mean += value;
                    std += value * value;
                }

                mean /= tensors.size();
                std = Math.sqrt(std / tensors.size() - mean * mean);

                means[f] = (float) mean;
                stds[f] = (float) Math.max(std, 1e-8);
            }

            Tensor meanTensor = Tensors.vector(means);
            Tensor stdTensor = Tensors.vector(stds);

            for (Tensor tensor : tensors) {
                tensor.sub(meanTensor).div(stdTensor);
            }
        }

        batchedInputs.clear();
        batchedLabels.clear();

        computeBatches();

        return this;
    }

    /**
     * Returns the next batch and advances the cursor.
     *
     * @return the next batch, or {@code null} if exhausted
     */
    public Batch nextBatch() {
        if (!hasNext()) {
            return null;
        }

        Tensor[] input = batchedInputs.get(cursor);
        Tensor[] label = batchedLabels.get(cursor);

        cursor++;

        return new Batch(input, label);
    }

    private void computeBatches() {
        int size = getSize();
        int index = 0;

        while (index < size) {
            int end = Math.min(index + batchSize, size);
            Batch batch = createBatch(index, end);

            batchedInputs.add(batch.first());
            batchedLabels.add(batch.second());

            index += batchSize;
        }
    }

    private Batch createBatch(int start, int end) {
        List<Sample> subSet = samples.subList(start, end);

        Sample first = subSet.getFirst();

        int inputCount = first.inputs().length;
        int labelCount = first.labels().length;

        List<List<Tensor>> mergedInputs = new ArrayList<>(inputCount);
        List<List<Tensor>> mergedLabels = new ArrayList<>();

        for (int i = 0; i < inputCount; i++) {
            mergedInputs.add(new ArrayList<>());
        }

        for (int i = 0; i < labelCount; i++) {
            mergedLabels.add(new ArrayList<>());
        }

        for (Sample sample : subSet) {
            Tensor[] inputs = sample.inputs();
            Tensor[] labels = sample.labels();

            for (int i = 0; i < inputs.length; i++) {
                mergedInputs.get(i).add(inputs[i]);
            }

            for (int i = 0; i < labels.length; i++) {
                mergedLabels.get(i).add(labels[i]);
            }
        }

        Tensor[] batchedInputTensors = new Tensor[inputCount];

        for (int i = 0; i < inputCount; i++) {
            Tensor merged = Tensors.mergeTensors(mergedInputs.get(i));
            batchedInputTensors[i] = device == null ? merged : GpuTensor.persistent(merged, device);
        }

        Tensor[] batchedLabelTensors = new Tensor[labelCount];

        for (int i = 0; i < labelCount; i++) {
            Tensor merged = Tensors.mergeTensors(mergedLabels.get(i));
            batchedLabelTensors[i] = device == null ? merged : GpuTensor.persistent(merged, device);
        }

        return new Batch(batchedInputTensors, batchedLabelTensors);
    }

    /**
     * Returns a deep copy of this source.
     *
     * @return the cloned source
     */
    @Override
    public ListDataSource clone() {
        try {
            ListDataSource clone = (ListDataSource) super.clone();

            clone.samples = new ArrayList<>();
            clone.batchSize = batchSize;

            for (Sample sample : samples) {
                clone.samples.add(sample.copy());
            }

            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns a copy staging batches on the given device.
     *
     * @return the device-backed source
     */
    public ListDataSource to(Device device) {
        return new ListDataSource(samples, device, batches, cursor, batchSize);
    }

    /**
     * Returns the total sample count.
     *
     * @return the number of samples
     */
    public int getSize() {
        return samples.size();
    }

    /**
     * Returns the underlying samples.
     *
     * @return the samples list
     */
    public List<Sample> getSamples() {
        return samples;
    }

    /**
     * Returns the batched inputs.
     *
     * @return the input batches
     */
    public List<Tensor[]> getBatchedInputs() {
        return batchedInputs;
    }

    /**
     * Returns the batched labels.
     *
     * @return the label batches
     */
    public List<Tensor[]> getBatchedLabels() {
        return batchedLabels;
    }

    /**
     * Returns the configured batch size.
     *
     * @return the batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Returns the total batch count.
     *
     * @return the number of batches
     */
    public int getBatches() {
        return batches;
    }

    /**
     * Returns the current batch cursor.
     *
     * @return the cursor position
     */
    public int getCursor() {
        return cursor;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Iterator<Sample> iterator() {
        return samples.iterator();
    }

    /**
     * Returns all batches as an iterable snapshot.
     *
     * @return the batches in order
     */
    public Iterable<Batch> batchIterator() {
        List<Batch> result = new ArrayList<>();
        int cursor = 0;

        while (cursor < batches) {
            Tensor[] input = batchedInputs.get(cursor);
            Tensor[] label = batchedLabels.get(cursor);

            cursor++;

            result.add(new Batch(input, label));
        }

        return result;
    }
}
