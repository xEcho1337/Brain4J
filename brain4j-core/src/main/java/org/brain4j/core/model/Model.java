package org.brain4j.core.model;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.Copyable;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.commons.Range;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.loss.impl.BinaryCrossEntropy;
import org.brain4j.math.tensor.Tensor;

/**
 * Defines the forward computation and structural introspection of a neural network.
 * Use it as the entry point for prediction, evaluation and device placement.
 *
 * @author xEcho1337
 */
public interface Model extends Copyable<Model> {

    /**
     * Runs a forward pass and returns the first output tensor.
     *
     * @return the first output tensor produced by the model
     */
    default Tensor predict(Tensor... inputs) {
        return predict(new StatesCache(), inputs)[0];
    }

    /**
     * Runs a forward pass using the given cache for intermediate states.
     *
     * @return all output tensors of the model
     */
    Tensor[] predict(StatesCache cache, Tensor... inputs);

    /**
     * Returns the device holding this model parameters.
     *
     * @return the associated device, or {@code null} for CPU
     */
    Device device();

    /**
     * Evaluates the model over the given dataset.
     *
     * @return an {@link EvaluationResult} containing loss and metrics
     */
    default EvaluationResult evaluate(ListDataSource dataSource, LossFunction lossFunction) {
        int classes = Math.max(2, dataSource.getSamples().getFirst().getLabel(0).elements());
        Map<Integer, Tensor> classifications = new HashMap<>();

        if (!lossFunction.isRegression()) {
            for (int i = 0; i < classes; i++) {
                classifications.put(i, Tensors.zeros(classes));
            }
        }

        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        dataSource.reset();

        while (dataSource.hasNext()) {
            Batch batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss, lossFunction);
        }

        return new EvaluationResult(totalLoss.get() / dataSource.getSize(), classes, classifications);
    }

    /**
     * Computes the average loss over the given dataset.
     *
     * @return the average loss over all samples
     */
    default double loss(ListDataSource dataSource, LossFunction lossFunction) {
        Map<Integer, Tensor> classifications = new HashMap<>();
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        dataSource.reset();

        while (dataSource.hasNext()) {
            Batch batch = dataSource.nextBatch();
            makeEvaluation(batch, classifications, totalLoss, lossFunction);
        }

        return totalLoss.get() / dataSource.getSize();
    }

    /**
     * Prints a formatted summary of the model architecture to the console.
     *
     * @throws IllegalStateException if the model has not been properly initialized
     */
    void summary();

    /**
     * Creates an independent copy of this model on the given device.
     *
     * @param device the target device ({@code null} for CPU)
     * @return an independent copy stored on the given device
     */
    Model fork(Device device);

    /**
     * Moves this model to the given device and returns it for chaining.
     *
     * @param device the device to move the model to ({@code null} for CPU)
     * @return this model, now stored on the given device
     */
    Model to(Device device);

    /**
     * Returns the layers composing this model, in order.
     *
     * @return an unmodifiable list of layers
     */
    List<Layer> getLayers();

    private void makeEvaluation(
        Batch batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss,
        LossFunction lossFunction
    ) {
        Tensor[] inputs = batch.first();
        Tensor[] labels = batch.second();

        Device device = device();

        if (device != null) {
            device.createResources();
        }

        StatesCache cache = new StatesCache(false);
        Tensor[] outputs = predict(cache, inputs);

        for (Tensor input : inputs) {
            int batchSize = input.shapeAt(0);

            for (int i = 0; i < outputs.length; i++) {
                Tensor output = outputs[i].to(null);
                Tensor label = labels[i].to(null);

                for (int b = 0; b < batchSize; b++) {
                    Range range = Range.point(b);

                    Tensor sampleOutput = output.slice(range).flatten();
                    Tensor sampleLabel = label.slice(range).flatten();

                    int predIndex = sampleOutput.argmax();
                    int targetIndex = sampleLabel.argmax();

                    if (sampleOutput.elements() == 1 && lossFunction instanceof BinaryCrossEntropy) {
                        predIndex = sampleOutput.get(0) > 0.5 ? 1 : 0;
                        targetIndex = (int) sampleLabel.get(0);
                    }

                    double loss = lossFunction.calculate(sampleLabel, sampleOutput);
                    totalLoss.updateAndGet(v -> v + loss);

                    Tensor predictions = classifications.get(targetIndex);

                    if (predictions != null) {
                        int pred = (int) predictions.get(predIndex);
                        predictions.set(pred + 1, predIndex);
                    }
                }
            }
        }

        if (device != null) {
            device.closeResources();
        }
    }
}
