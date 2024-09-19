package com.github.echo.training;

import com.github.echo.network.NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractModel {

    protected final NeuralNetwork network;
    protected int batches = 1;
    protected double learningRate;
    protected double totalError;

    public AbstractModel(NeuralNetwork network) {
        this.network = network;
    }

    /**
     * Trains the model using the given data set and precision.
     *
     * @param  dataSet   the data set to train the model on
     * @param  precision the desired precision for the training
     */
    public abstract void train(DataSet dataSet, double precision);

    /**
     * Iterates one time over the given data set.
     *
     * @param  dataSet   the data set to iterate over
     */
    public abstract void iterate(DataSet dataSet);

    /**
     * Returns the NeuralNetwork object associated with this AbstractModel.
     *
     * @return the NeuralNetwork object
     */
    public NeuralNetwork network() {
        return network;
    }

    /**
     * Returns the number of batches for the model.
     *
     * @return the number of batches
     */
    public int getBatches() {
        return batches;
    }

    /**
     * Sets the number of batches for the model.
     *
     * @param  batches  the number of batches to set
     */
    public void setBatches(int batches) {
        this.batches = batches;
    }

    /**
     * Returns the learning rate for the model.
     *
     * @return the learning rate value
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets the learning rate for the model.
     *
     * @param  learningRate  the new learning rate value
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Returns the total error of the network.
     *
     * @return the total error
     */
    public double getTotalError() {
        return totalError;
    }

    protected static <T> List<List<T>> partition(List<T> list, int batchSize) {
        List<List<T>> batches = new ArrayList<>();
        int totalSize = list.size();

        for (int i = 0; i < totalSize; i += batchSize) {
            batches.add(list.subList(i, Math.min(i + batchSize, totalSize)));
        }

        return batches;
    }
}
