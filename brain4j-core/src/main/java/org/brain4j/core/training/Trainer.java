package org.brain4j.core.training;

import java.util.List;

import org.brain4j.core.model.Model;
import org.brain4j.core.monitor.Monitor;
import org.brain4j.math.commons.Batch;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Tensor;

/**
 * Drives model optimization over datasets.
 * Use it to run synchronous or background training with monitors.
 */
public interface Trainer {

    /**
     * Creates a trainer without monitors.
     *
     * @return a new trainer
     */
    static DefaultTrainer of(Model model, TrainingConfig config) {
        return new DefaultTrainer(model, config, List.of());
    }

    /**
     * Creates a trainer with the given monitors.
     *
     * @return a new trainer
     */
    static DefaultTrainer of(Model model, TrainingConfig config, List<Monitor> monitors) {
        return new DefaultTrainer(model, config, monitors);
    }

    /**
     * Returns the attached monitor of the given type.
     *
     * @return the matching monitor, or {@code null} if missing
     */
    <T extends Monitor> T getMonitor(Class<T> monitorClass);

    /**
     * Attaches a monitor to this trainer.
     */
    <T extends Monitor> void attach(T monitor);

    /**
     * Starts training in a background thread.
     *
     * @return the worker thread
     */
    Thread start(ListDataSource dataSource, int epochs);

    /**
     * Trains synchronously for the given epochs.
     */
    void fit(ListDataSource dataSource, int epochs);

    /**
     * Pauses an ongoing training run.
     */
    void pause();

    /**
     * Resumes a paused training run.
     */
    void resume();

    /**
     * Stops an ongoing training run.
     */
    void stop();

    /**
     * Checks whether training is running.
     *
     * @return {@code true} if training is active
     */
    boolean isTraining();

    /**
     * Checks whether training is paused.
     *
     * @return {@code true} if paused
     */
    boolean isPaused();

    /**
     * Runs a forward pass and caches intermediate states.
     *
     * @return the output tensors
     */
    Tensor[] forward(StatesCache cache, Tensor[] inputs);

    /**
     * Runs backpropagation for the given batch outputs.
     */
    void backward(StatesCache cache, Batch batch, Tensor[] outputs);

    /**
     * Clears gradients of all model parameters.
     */
    void resetGrad();

    /**
     * Returns the zero-based current epoch.
     *
     * @return the current epoch
     */
    int currentEpoch();

    /**
     * Returns the zero-based current batch.
     *
     * @return the current batch
     */
    int currentBatch();

    /**
     * Returns the total epoch count.
     *
     * @return the total epochs
     */
    int totalEpochs();

    /**
     * Returns the total batch count per epoch.
     *
     * @return the total batches
     */
    int totalBatches();

    /**
     * Returns the training configuration.
     *
     * @return the active config
     */
    TrainingConfig config();

    /**
     * Returns the trained model.
     *
     * @return the model
     */
    Model model();

    /**
     * Returns the attached monitors.
     *
     * @return the monitors
     */
    List<Monitor> monitors();
}
