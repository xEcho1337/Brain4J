package org.brain4j.core.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.math.Copyable;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;

/**
 * Collects input shape and blocks before compiling a model.
 * Use it to assemble layers fluently and then compile a {@link Sequential} model.
 */
public class ModelLayout implements ModelBlock, Copyable<ModelLayout> {

    private final Shape inputShape;
    private final List<ModelBlock> components = new ArrayList<>();

    private ModelLayout(Shape inputShape) {
        if (inputShape == null) {
            throw Commons.illegalArgument("Input shape must not be null!");
        }

        this.inputShape = inputShape;
    }

    /**
     * Creates a new layout builder for a model with the given input shape.
     *
     * @param input the input shape
     * @return a new builder
     */
    public static Builder stack(Shape input) {
        return new Builder(input);
    }

    /**
     * Creates a layout from the given input shape and components.
     *
     * @param inputShape the model input shape (not null)
     * @return a new model layout instance
     */
    public static ModelLayout of(Shape inputShape, List<ModelBlock> components) {
        if (components == null) {
            throw Commons.illegalArgument("Component list must not be null!");
        }

        ModelLayout layout = new ModelLayout(inputShape);
        layout.components.addAll(components);

        return layout;
    }

    /**
     * Creates specs from the given input shape and components.
     *
     * @param inputShape the model input shape (not null)
     * @return a new mutable specs instance
     */
    public static ModelLayout of(Shape inputShape, ModelBlock... components) {
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
        return new Sequential(this, null, seed);
    }

    /**
     * Returns a view of the components in this specification.
     *
     * @return an unmodifiable list of components
     */
    public List<ModelBlock> components() {
        return Collections.unmodifiableList(components);
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
    public ModelLayout copy() {
        ModelLayout copy = new ModelLayout(inputShape.copy());
        copy.components.addAll(components);
        return copy;
    }

    public static class Builder {

        private final Shape inputShape;
        private final List<ModelBlock> components;

        public Builder(Shape inputShape) {
            this.inputShape = inputShape;
            this.components = new ArrayList<>();
        }

        /**
         * Adds a component to the end of the builder.
         *
         * @param component the block to add (not null)
         * @return this builder for chaining
         * @throws NullPointerException if the component is null
         */
        public Builder add(ModelBlock component) {
            Objects.requireNonNull(component, "Component must not be null!");
            components.add(component);

            return this;
        }

        /**
         * Adds one or multiple components to the end of the builder.
         *
         * @param components the components to add
         * @return this builder for chaining
         * @throws NullPointerException if any of the components is null
         */
        public Builder addAll(ModelBlock... components) {
            Objects.requireNonNull(components, "Component list must not be null!");

            for (ModelBlock component : components) {
                Objects.requireNonNull(component, "Component must not be null!");
            }

            this.components.addAll(List.of(components));

            return this;
        }

        /**
         * Compiles the builder into an immutable specs instance.
         * @return the compiled specs
         */
        public ModelLayout build() {
            return ModelLayout.of(inputShape, components);
        }
    }
}
