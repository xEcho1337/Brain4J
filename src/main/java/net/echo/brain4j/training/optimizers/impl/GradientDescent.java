package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GradientDescent extends Optimizer {

    private final Map<Synapse, Double> gradientMap = new HashMap<>();

    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    @Override
    public void update(Synapse synapse) {
        // We don't use this
    }

    @Override
    public void postIteration(List<Layer> layers) {
    }

    @Override
    public void postFit(List<Layer> layers) {
        for (Map.Entry<Synapse, Double> entry : gradientMap.entrySet()) {
            Synapse synapse = entry.getKey();

            double gradientSum = entry.getValue();

            synapse.setWeight(synapse.getWeight() + learningRate * gradientSum);
        }

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getDelta();
                neuron.setBias(neuron.getBias() + deltaBias);
            }
        }

        gradientMap.clear();
    }

    @Override
    public void applyGradientStep(Layer layer, Neuron neuron, Synapse synapse) {
        double gradient = calculateGradient(layer, synapse, neuron);
        gradientMap.put(synapse, gradientMap.getOrDefault(synapse, 0.0) + gradient);
    }

    /**
     * Calculate the gradient for a synapse based on the delta and the value of the input.
     *
     * @param synapse the synapse
     * @param neuron  the neuron
     * @return the calculated gradient
     */
    private double calculateGradient(Layer layer, Synapse synapse, Neuron neuron) {
        double output = neuron.getValue();

        double error = clipGradient(synapse.getWeight() * synapse.getOutputNeuron().getDelta());
        double delta = clipGradient(error * layer.getActivation().getFunction().getDerivative(output));

        neuron.setDelta(delta);

        return clipGradient(delta * synapse.getInputNeuron().getValue());
    }
}