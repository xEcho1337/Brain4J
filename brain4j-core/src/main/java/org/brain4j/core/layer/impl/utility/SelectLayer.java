package org.brain4j.core.layer.impl.utility;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Forwards only the input at the selected index.
 * Use it to pick one branch out of several graph inputs.
 */
public class SelectLayer extends Layer {

    /**
     * Holds the selected input index.
     */
    public record Config(int index) {}

    protected final Config config;

    /**
     * Create a layer selecting the given input index.
     */
    public SelectLayer(int index) {
        this(new Config(index));
    }

    /**
     * Create a layer with the given config.
     */
    public SelectLayer(Config config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.isEmpty()) {
            throw Commons.illegalArgument("Layer requires at least 1 input but 0 were given!");
        }

        if (config.index < 0 || config.index >= inputShapes.size()) {
            throw Commons.illegalArgument("Selection index %s is out of range (size=%s)", config.index, inputShapes.size());
        }

        return List.of(inputShapes.get(config.index));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return tensors(inputs[config.index]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        return new SelectLayer(config);
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }
}
