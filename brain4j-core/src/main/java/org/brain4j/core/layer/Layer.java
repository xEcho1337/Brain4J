package org.brain4j.core.layer;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

import org.brain4j.core.model.ModelBlock;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Copyable;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.clipper.impl.HardClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.WeightInit;

/**
 * Parametric transformation applied to input tensors.
 * Use it as the building block for sequential and graph models.
 */
public abstract class Layer implements Copyable<Layer>, ModelBlock {

    protected final Map<String, Tensor> parameters;

    protected GradientClipper clipper;
    protected Activation activation;
    protected WeightInit weightInit;
    protected boolean frozen;

    public Layer() {
        this(new Linear());
    }

    /**
     * Creates a layer with the given activation.
     */
    public Layer(Activation activation) {
        this.parameters = new HashMap<>();
        this.clipper = new HardClipper(5);
        this.activation = activation;
        this.weightInit = activation.defaultWeightInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void appendTo(List<Layer> layers) {
        layers.add(this);
    }

    /**
     * Allocates parameters from the given input shapes.
     */
    public abstract void build(List<Shape> inputShapes);

    /**
     * Initializes parameters from the given input shapes and RNG.
     */
    public abstract void initWeights(List<Shape> inputShapes, RandomGenerator rng);

    /**
     * Infers output shapes from the given input shapes.
     *
     * @return the inferred output shapes
     */
    public abstract List<Shape> inferOutputShapes(List<Shape> inputShapes);

    /**
     * Runs a forward pass on a single input.
     *
     * @return the output tensor
     */
    public Tensor forward(StatesCache cache, Tensor input) {
        return forward(cache, new Tensor[] { input })[0];
    }

    /**
     * Runs a forward pass on the given inputs.
     *
     * @return the output tensors
     */
    public abstract Tensor[] forward(StatesCache cache, Tensor... inputs);

    /**
     * Wraps the given values into an output array.
     *
     * @return the wrapped values
     */
    protected Tensor[] tensors(Tensor... values) {
        return values;
    }

    /**
     * Enables gradient tracking on all parameters.
     */
    public void initAutoGrad() {
        for (Tensor param : parameters.values()) {
            param.withGrad();
        }
    }

    /**
     * Fills the named parameter with values from the weight initializer.
     *
     * @param id the parameter name (must exist)
     * @param rng the random source
     * @param input the fan-in size
     * @param output the fan-out size
     */
    public void generateWeights(String id, RandomGenerator rng, int input, int output) {
        Tensor param = parameters.get(id);

        if (param == null) {
            throw Commons.illegalArgument("No parameter with id '%s' was found!", id);
        }

        // Weight init is intentionally sequential to keep runs reproducible.
        float[] data = param.data();

        for (int i = 0; i < data.length; i++) {
            data[i] = (float) weightInit.generate(rng, input, output);
        }
    }

    /**
     * Copies parameters into the given layer, preserving gradient flags.
     *
     * @param other the target layer
     */
    public void copyParameters(Layer other) {
        Map<String, Tensor> newParameters = new HashMap<>();
        parameters.forEach((k, v) -> {
            Tensor copy = copyStable(v);

            if (v.usesGrad()) {
                copy.withGrad();
            } else {
                copy.noGrad();
            }

            newParameters.put(k, copy);
        });

        other.parameters.clear();
        other.parameters.putAll(newParameters);
        other.frozen = this.frozen;
    }

    /**
     * Moves parameters to the given device.
     *
     * @param device the target device ({@code null} for CPU)
     * @return this layer for chaining
     */
    public Layer to(Device device) {
        Map<String, Tensor> newParameters = new HashMap<>();
        parameters.forEach((k, v) -> {
            Tensor moved = device == null ? v.to(null) : copyStable(v, device);

            if (v.usesGrad()) {
                moved.withGrad();
            } else {
                moved.noGrad();
            }

            newParameters.put(k, moved);
        });

        parameters.clear();
        parameters.putAll(newParameters);
        return this;
    }

    private static Tensor copyStable(Tensor tensor) {
        if (tensor instanceof GpuTensor gpuTensor) {
            return GpuTensor.persistent(tensor, gpuTensor.device());
        }

        return tensor.copy();
    }

    private static Tensor copyStable(Tensor tensor, Device device) {
        if (tensor instanceof GpuTensor gpuTensor
            && gpuTensor.device() == device
            && gpuTensor.isPersistent()) {
            return tensor;
        }

        return GpuTensor.persistent(tensor, device);
    }

    /**
     * Connects this layer to the given inputs.
     *
     * @return the output node
     */
    public Node apply(Node... inputs) {
        return new Node(this, List.of(inputs));
    }

    /**
     * Registers a trainable parameter.
     *
     * @param name the parameter name (not null)
     * @param param the parameter tensor
     */
    public void registerParam(String name, Tensor param) {
        parameters.put(name, param);
    }

    /**
     * Returns the parameter with the given name.
     *
     * @return the parameter, or {@code null} if missing
     */
    public Tensor param(String name) {
        return parameters.get(name);
    }

    /**
     * Clears gradients of all parameters.
     */
    public void resetGrad() {
        for (Tensor parameter : parameters.values()) {
            parameter.zeroGrad();
        }
    }

    /**
     * Freezes parameters by disabling gradient tracking.
     *
     * @return this layer for chaining
     */
    public Layer freeze() {
        frozen = true;
        parameters.replaceAll((k, v) -> v.noGrad());

        return this;
    }

    /**
     * Unfreezes parameters by enabling gradient tracking.
     *
     * @return this layer for chaining
     */
    public Layer unfreeze() {
        frozen = false;
        parameters.replaceAll((k, v) -> v.withGrad());

        return this;
    }

    /**
     * Counts elements of parameters with gradient tracking enabled.
     *
     * @return the trainable parameter count
     */
    public int calculateTrainableParams() {
        return parameters.values()
            .stream()
            .filter(Tensor::usesGrad)
            .mapToInt(Tensor::elements)
            .sum();
    }

    /**
     * Counts elements of all parameters.
     *
     * @return the total parameter count
     */
    public int calculateTotalParameters() {
        return parameters.values()
            .stream()
            .mapToInt(Tensor::elements)
            .sum();
    }

    /**
     * Returns all registered parameters.
     *
     * @return the parameter map
     */
    public Map<String, Tensor> parameters() {
        return parameters;
    }

    /**
     * Returns the gradient clipper.
     *
     * @return the active clipper
     */
    public GradientClipper clipper() {
        return clipper;
    }

    /**
     * Replaces the gradient clipper.
     *
     * @return this layer for chaining
     */
    public Layer clipper(GradientClipper clipper) {
        this.clipper = clipper;
        return this;
    }

    /**
     * Returns the activation function.
     *
     * @return the active activation
     */
    public Activation activation() {
        return activation;
    }

    /**
     * Replaces the activation function.
     *
     * @return this layer for chaining
     */
    public Layer activation(Activation activation) {
        this.activation = activation;
        return this;
    }

    /**
     * Returns the weight initializer.
     *
     * @return the active initializer
     */
    public WeightInit weightInit() {
        return weightInit;
    }

    /**
     * Replaces the weight initializer.
     *
     * @return this layer for chaining
     */
    public Layer weightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
        return this;
    }

    /**
     * Checks whether parameters are frozen.
     *
     * @return {@code true} if gradient tracking is disabled
     */
    public boolean frozen() {
        return frozen;
    }

    /**
     * Applies optimizer and updater steps to parameter gradients using a single sample.
     */
    public void backward(Updater updater, Optimizer optimizer) {
        backward(updater, optimizer, 1);
    }

    /**
     * Applies optimizer and updater steps to parameter gradients.
     *
     * @param samples number of samples in the current batch
     */
    public void backward(Updater updater, Optimizer optimizer, int samples) {
        for (Tensor parameter : parameters.values()) {
            Tensor grad = parameter.grad();

            if (grad == null) {
                continue;
            }

            while (grad.rank() > parameter.rank()) {
                grad = grad.sum(0, false);
            }

            Tensor optimized = optimizer.step(parameter, grad, samples);

            clipper.clip(optimized);
            updater.change(parameter, optimized);
        }
    }
}
