package net.echo.brain4j.layer;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

@JsonAdapter(LayerAdapter.class)
public class Layer {

    private final List<Neuron> neurons = new ArrayList<>();
    private final List<Synapse> synapses = new ArrayList<>();
    private final Activations activation;

    public Layer(int input, Activations activation) {
        for (int i = 0; i < input; i++) {
            neurons.add(new Neuron());
        }

        this.activation = activation;
    }

    public void connectAll(Layer nextLayer, double bound) {
        for (Neuron neuron : neurons) {
            for (Neuron nextNeuron : nextLayer.neurons) {
                Synapse synapse = new Synapse(neuron, nextNeuron, bound);

                neuron.addSynapse(synapse);

                synapses.add(synapse);
            }
        }
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public List<Synapse> getSynapses() {
        return synapses;
    }

    public Activations getActivation() {
        return activation;
    }

    public Neuron getNeuronAt(int i) {
        return neurons.get(i);
    }

    public void applyFunction(Layer previous) {
        Activation function = activation.getFunction();

        function.apply(neurons);
    }

    public int getTotalParams() {
        return synapses.size();
    }

    public void setInput(Vector input) {
        if (input.size() != neurons.size()) {
            throw new IllegalArgumentException("Input size does not match model's input dimension! (Input != Expected) " +
                    input.size() + " != " + neurons.size());
        }

        for (int i = 0; i < input.size(); i++) {
            neurons.get(i).setValue(input.get(i));
        }
    }

    public void activate() {
        for (Synapse synapse : synapses) {
            Neuron inputNeuron = synapse.getInputNeuron();
            Neuron outputNeuron = synapse.getOutputNeuron();

            outputNeuron.setValue(outputNeuron.getValue() + inputNeuron.getValue() * synapse.getWeight());
        }
    }

    public void activate(Vector input) {
        for (int i = 0; i < neurons.size(); i++) {
            Neuron inputNeuron = neurons.get(i);

            for (Synapse synapse : inputNeuron.getSynapses()) {
                Neuron outputNeuron = synapse.getOutputNeuron();

                outputNeuron.setValue(outputNeuron.getValue() + input.get(i) * synapse.getWeight());
            }
        }
    }

    public Vector getVector() {
        Vector values = new Vector(neurons.size());

        for (int i = 0; i < neurons.size(); i++) {
            values.set(i, neurons.get(i).getValue());
        }

        return values;
    }

    public void propagate(Optimizer optimizer) {
        for (Neuron neuron : neurons) {
            for (Synapse synapse : neuron.getSynapses()) {
                optimizer.applyGradientStep(this, neuron, synapse);
            }
        }
    }
}
