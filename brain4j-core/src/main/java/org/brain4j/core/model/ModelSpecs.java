package org.brain4j.core.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.Copyable;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;

/**
 * Collects input shape and blocks before compiling a model.
 * Use it to assemble layers fluently and then compile a {@link Sequential} model.
 */
public class ModelSpecs implements ModelBlock, Copyable<ModelSpecs> {

    private final Shape inputShape;
    private final List<ModelBlock> components = new ArrayList<>();
    private boolean frozen = false;

    private ModelSpecs(Shape inputShape) {
        if (inputShape == null) {
            throw Commons.illegalArgument("Input shape must not be null!");
        }

        this.inputShape = inputShape;
    }

    /**
     * Creates specs from the given input shape and components.
     *
     * @param inputShape the model input shape (not null)
     * @return a new mutable specs instance
     */
    public static ModelSpecs of(Shape inputShape, List<ModelBlock> components) {
        if (components == null) {
            throw Commons.illegalArgument("Component list must not be null!");
        }

        ModelSpecs specs = new ModelSpecs(inputShape);
        specs.components.addAll(components);

        return specs;
    }

    /**
     * Creates specs from the given input shape and components.
     *
     * @param inputShape the model input shape (not null)
     * @return a new mutable specs instance
     */
    public static ModelSpecs of(Shape inputShape, ModelBlock... components) {
        if (components == null) {
            throw Commons.illegalArgument("Component list must not be null!");
        }

        return of(inputShape, List.of(components));
    }

    /**
     * Returns the declared model input shape.
     *
     * @return the input shape
     */
    public Shape inputShape() {
        return inputShape;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void appendTo(List<Layer> layers) {
        for (ModelBlock component : components) {
            component.appendTo(layers);
        }
    }

    /**
     * Adds a block to these specs.
     *
     * @param component the block to add (not null)
     * @return this specs for chaining
     * @throws IllegalArgumentException if already compiled
     */
    public ModelSpecs add(ModelBlock component) {
        if (frozen) {
            throw new IllegalArgumentException("ModelSpecs has been compiled and cannot be modified! Consider checking out clone().");
        }

        if (component == null) {
            throw Commons.illegalArgument("Component must not be null!");
        }

        components.add(component);
        return this;
    }

    /**
     * Compiles these specs into a model with a random seed.
     *
     * @return the compiled model
     */
    public Sequential compile() {
        return compile((int) (System.currentTimeMillis() % 1_000_000_000));
    }

    /**
     * Compiles these specs into a model with the given seed.
     *
     * @return the compiled model
     */
    public Sequential compile(int seed) {
        this.frozen = true;
        return new Sequential(this, null, seed);
    }

    /**
     * Returns the blocks added to these specs.
     *
     * @return an unmodifiable view once compiled, otherwise the live list
     */
    public List<ModelBlock> getComponents() {
        if (frozen) {
            return Collections.unmodifiableList(components);
        }

        return components;
    }

    /**
     * Flattens all blocks into an ordered layer list.
     *
     * @return the flattened layers
     */
    public List<Layer> buildLayerList() {
        List<Layer> flat = new ArrayList<>();
        appendTo(flat);
        return flat;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ModelSpecs copy() {
        ModelSpecs copy = new ModelSpecs(inputShape.copy());
        copy.components.addAll(components);
        return copy;
    }
}
