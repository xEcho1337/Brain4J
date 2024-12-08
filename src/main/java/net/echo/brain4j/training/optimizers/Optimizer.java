package net.echo.brain4j.training.optimizers;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;

import java.util.List;

/**
 * Abstract class for optimization algorithms.
 */
@JsonAdapter(OptimizerAdapter.class)
public abstract class Optimizer {

    private static final double GRADIENT_CLIP = 5.0;
    protected double learningRate;

    /**
     * Initializes the optimizer with a specified learning rate.
     *
     * @param learningRate the learning rate
     */
    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Called after the network has been compiled and all the synapses have been initialized.
     */
    public void postInitialize() {
    }

    /**
     * Gets the current learning rate.
     *
     * @return learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets a new learning rate.
     *
     * @param learningRate the new learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Updates the given synapse based on the optimization algorithm.
     *
     * @param synapse the synapse to update
     */
    public abstract void update(Synapse synapse);

    /**
     * Called after a sample has been iterated.
     *
     * @param layers the layers of the model
     */
    public abstract void postIteration(List<Layer> layers);

    /**
     * Called after all samples in the dataset have been iterated.
     *
     * @param layers the layers of the model
     */
    public void postFit(List<Layer> layers) {
    }

    /**
     * Updates the given synapse based on the optimization algorithm.
     *
     * @param layer    the layer of the neuron
     * @param neuron   the neuron connected to the synapse
     * @param synapse  the synapse involved
     */
    public void applyGradientStep(Layer layer, Neuron neuron, Synapse synapse) {
        double output = neuron.getValue();

        double error = clipGradient(synapse.getWeight() * synapse.getOutputNeuron().getDelta());
        double delta = clipGradient(error * layer.getActivation().getFunction().getDerivative(output));

        double weightChange = clipGradient(delta * synapse.getInputNeuron().getValue());

        neuron.setDelta(delta);
        synapse.setWeight(synapse.getWeight() + weightChange);
    }

    /**
     * Clips the gradient to avoid gradient explosion.
     *
     * @param gradient the gradient
     * @return the clipped gradient
     */
    public double clipGradient(double gradient) {
        return Math.max(Math.min(gradient, GRADIENT_CLIP), -GRADIENT_CLIP);
    }
}