package org.brain4j.core.model.impl;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.IntStream;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.Model;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.Tensor;

/**
 * Directed acyclic graph of layers with arbitrary topology.
 * Use it for multi-input, multi-output or branching architectures.
 */
public class Graph implements Model {

    private final List<Node> input;
    private final List<Node> output;
    private final List<Node> topology;
    private final Device device;
    private final int seed;

    /**
     * Creates a graph ending at the given outputs on CPU.
     *
     * @param output the output nodes (not null)
     * @param device the target device ({@code null} for CPU)
     * @param seed the weight initialization seed
     */
    protected Graph(List<Node> output, Device device, int seed) {
        this(output, device, seed, false);
    }

    private Graph(List<Node> output, Device device, int seed, boolean isCopy) {
        this.device = device;
        this.output = output;
        this.seed = seed;
        this.input = new ArrayList<>();
        this.topology = new ArrayList<>();

        Set<Node> visited = new HashSet<>();

        for (Node outNode : output) {
            dfs(outNode, visited);
        }

        if (isCopy) {
            return;
        }

        for (Node in : input) {
            in.build();
        }

        for (Node topologyNode : topology) {
            topologyNode.build();
        }

        IntStream.range(0, topology.size()).parallel().forEach(i -> {
            Node topologyNode = topology.get(i);
            topologyNode.initWeights(seed + i);
        });
    }

    /**
     * Creates a graph ending at the given outputs with a default seed.
     *
     * @return a new graph on CPU
     */
    public static Graph of(Node... output) {
        return new Graph(List.of(output), null, 42);
    }

    /**
     * Creates a graph ending at the given outputs with the given seed.
     *
     * @return a new graph on CPU
     */
    public static Graph of(int seed, Node... output) {
        return new Graph(List.of(output), null, seed);
    }

    /**
     * Creates a graph ending at the given outputs on the given device.
     *
     * @param device the target device ({@code null} for CPU)
     * @return a new graph
     */
    public static Graph of(int seed, Device device, Node... output) {
        return new Graph(List.of(output), device, seed);
    }

    private void dfs(Node node, Set<Node> visited) {
        if (visited.contains(node)) {
            return;
        }

        visited.add(node);

        for (Node input : node.inputs()) {
            dfs(input, visited);
        }

        if (node.inputs().isEmpty()) {
            input.add(node);
        } else {
            topology.add(node);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] predict(StatesCache cache, Tensor... inputs) {
        if (input.size() != inputs.length) {
            throw Commons.illegalArgument("DAG expects %s inputs but %s were given!", input.size(), inputs.length);
        }

        Map<Node, Tensor[]> outputs = new HashMap<>();

        for (int i = 0; i < input.size(); i++) {
            Tensor source = inputs[i];
            outputs.put(input.get(i), new Tensor[]{cache.isTraining() ? source.withGrad() : source});
        }

        for (Node node : topology) {
            node.forward(cache, outputs);
        }

        List<Tensor> finalOutputs = new ArrayList<>();

        for (Node outNode : output) {
            Tensor[] nodeOutputs = outputs.get(outNode);
            Collections.addAll(finalOutputs, nodeOutputs);
        }

        return finalOutputs.toArray(new Tensor[0]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Device device() {
        return device;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void summary() {
        throw Commons.illegalState("summary() is not supported in Graph impl. Use export(GraphExporter) instead.");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Graph fork(Device device) {
        Map<Node, Node> cache = new HashMap<>();
        List<Node> copy = output.stream().map(n -> n.copy(cache)).toList();

        for (Node n : cache.values()) {
            n.layer().to(device);
        }

        return new Graph(copy, device, seed, true);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Graph to(Device device) {
        if (Objects.equals(this.device, device)) {
            return this;
        }

        for (Node node : topology) {
            node.layer().to(device);
        }

        for (Node node : input) {
            node.layer().to(device);
        }

        return new Graph(output, device, seed, true);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Graph copy() {
        Map<Node, Node> cache = new HashMap<>();
        List<Node> copy = output.stream().map(n -> n.copy(cache)).toList();
        return new Graph(copy, device, seed, true);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Layer> getLayers() {
        return topology.stream().map(Node::layer).toList();
    }

    /**
     * Returns the input nodes of this graph.
     *
     * @return the input nodes
     */
    public List<Node> input() {
        return input;
    }

    /**
     * Returns the output nodes of this graph.
     *
     * @return the output nodes
     */
    public List<Node> output() {
        return output;
    }

    /**
     * Returns the topologically sorted inner nodes.
     *
     * @return the topology nodes
     */
    public List<Node> topology() {
        return topology;
    }

    /**
     * Returns the weight initialization seed.
     *
     * @return the seed
     */
    public int seed() {
        return seed;
    }
}
