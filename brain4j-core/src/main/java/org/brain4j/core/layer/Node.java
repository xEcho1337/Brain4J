package org.brain4j.core.layer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

import org.brain4j.core.importing.io.LayerIO;
import org.brain4j.math.Copyable;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Graph vertex binding a layer to its inputs.
 * Use it to wire layers into branching or merging topologies.
 */
public class Node implements Copyable<Node> {

    private final Layer layer;
    private final List<Node> inputs;
    private List<Shape> outputShapes;

    /**
     * Creates a node applying the given layer to the given inputs.
     */
    public Node(Layer layer, List<Node> inputs) {
        this.layer = layer;
        this.inputs = inputs;
    }

    /**
     * Creates a source node emitting the given shape.
     *
     * @return the input node
     */
    public static Node input(Shape shape) {
        return new Node(new Source(shape), List.of());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Node copy() {
        return copy(new HashMap<>());
    }

    /**
     * Copies this node and its ancestors, reusing already copied nodes.
     *
     * @return the copied node
     */
    public Node copy(Map<Node, Node> cache) {
        if (cache.containsKey(this)) {
            return cache.get(this);
        }

        Node copy = new Node(layer.copy(), new ArrayList<>());
        cache.put(this, copy);

        for (Node in : inputs) {
            copy.inputs.add(in.copy(cache));
        }

        copy.outputShapes = this.outputShapes == null ? null : new ArrayList<>(this.outputShapes);
        return copy;
    }

    /**
     * Infers output shapes and allocates layer parameters.
     */
    public void build() {
        List<Shape> inputShapes = inferInputShapes();
        this.outputShapes = layer.inferOutputShapes(inputShapes);

        if (layer.frozen()) {
            return;
        }

        layer.build(inputShapes);
    }

    /**
     * Initializes layer weights with the given seed.
     */
    public void initWeights(int seed) {
        List<Shape> inputShapes = inferInputShapes();
        RandomGenerator rng = new SplittableRandom(seed);

        if (layer.frozen()) {
            return;
        }

        layer.initWeights(inputShapes, rng);
        layer.initAutoGrad();
    }

    /**
     * Collects output shapes of all input nodes.
     *
     * @return the flattened input shapes
     */
    public List<Shape> inferInputShapes() {
        return inputs.stream()
            .flatMap(n -> n.outputShapes().stream())
            .toList();
    }

    /**
     * Runs this node forward and memoizes outputs.
     *
     * @return the output tensors
     */
    public Tensor[] forward(StatesCache cache, Map<Node, Tensor[]> computed) {
        List<Tensor> tensors = new ArrayList<>();

        for (Node input : inputs) {
            Tensor[] inTensors = computed.get(input);
            tensors.addAll(Arrays.asList(inTensors));
        }

        Tensor[] out = layer.forward(cache, tensors.toArray(Tensor[]::new));
        computed.put(this, out);

        return out;
    }

    /**
     * Returns the layer executed by this node.
     *
     * @return the layer
     */
    public Layer layer() {
        return layer;
    }

    /**
     * Returns the input nodes.
     *
     * @return the inputs
     */
    public List<Node> inputs() {
        return inputs;
    }

    /**
     * Returns the inferred output shapes.
     *
     * @return the output shapes, or {@code null} if not built
     */
    public List<Shape> outputShapes() {
        return outputShapes;
    }

    /**
     * Returns the registered codec type of the layer.
     *
     * @return the layer type name
     */
    public String name() {
        return LayerIO.LAYER_CODECS.get(layer.getClass()).type();
    }

    /**
     * Source node validating external inputs against a fixed shape.
     */
    private static final class Source extends Layer {
        private final Shape shape;

        private Source(Shape shape) {
            this.shape = shape;
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
            return List.of(shape);
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Tensor[] forward(StatesCache cache, Tensor... inputs) {
            for (Tensor input : inputs) {
                if (validInput(input)) {
                    continue;
                }

                throw Commons.illegalArgument("Input must have shape %s! Got: %s",
                    Arrays.toString(shape.dims()), Arrays.toString(input.shape()));
            }

            return inputs;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public Layer copy() {
            return new Source(shape.copy());
        }

        private boolean validInput(Tensor input) {
            if (input == null) {
                return false;
            }

            int[] inputShape = input.shape();
            int[] targetShape = shape.dims();

            if (inputShape.length - 1 > targetShape.length) {
                return false;
            }

            int offset = inputShape.length - targetShape.length;

            for (int i = 0; i < targetShape.length; i++) {
                if (inputShape[i + offset] != targetShape[i]) {
                    return false;
                }
            }

            return true;
        }
    }
}
