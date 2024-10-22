package net.echo.brain4j.model;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.InitializationType;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;

import java.util.List;

/**
 * Interface for a neural network model, defining methods for compilation, training, and prediction.
 */
public interface Model {

    /**
     * Initializes the model and layers.
     *
     * @param type      initialization method
     * @param function  loss function for error assessment
     * @param optimizer optimization algorithm
     */
    void compile(InitializationType type, LossFunctions function, Optimizer optimizer);

    /**
     * Trains the model for one epoch.
     *
     * @param set dataset for training
     */
    void fit(DataSet set);

    /**
     * Evaluates the model on the given dataset.
     *
     * @param set dataset for testing
     * @return the error of the model
     */
    double evaluate(DataSet set);

    /**
     * Predicts output for given input.
     *
     * @param input input data
     * @return predicted outputs
     */
    double[] predict(double... input);

    /**
     * Gets the model's loss function.
     *
     * @return loss function
     */
    LossFunction getLossFunction();

    /**
     * Retrieves layers of the network.
     *
     * @return list of layers
     */
    List<Layer> getLayers();

    /**
     * Generates model statistics.
     *
     * @return model stats
     */
    String getStats();

    /**
     * Loads a model from a file.
     *
     * @param path path to model file
     */
    void load(String path);

    /**
     * Saves the model to a file.
     *
     * @param path path to save model
     */
    void save(String path);

    /**
     * Adds a layer to the network.
     *
     * @param layer the layer to add
     */
    void add(Layer layer);
}