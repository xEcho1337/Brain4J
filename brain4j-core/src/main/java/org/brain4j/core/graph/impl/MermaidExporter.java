package org.brain4j.core.graph.impl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.brain4j.core.graph.GraphExporter;
import org.brain4j.core.importing.format.impl.OnnxFormat;
import org.brain4j.core.importing.io.OnnxIO;
import org.brain4j.core.layer.Node;
import org.brain4j.core.model.impl.Graph;
import org.brain4j.core.utils.CodeWriter;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Exports a graph model to Mermaid flowchart syntax.
 * Use it to visualize architectures in docs or debugging tools.
 */
public class MermaidExporter implements GraphExporter {

    private final Direction direction;
    private final boolean detailed;

    /**
     * Create an exporter with top-down layout and no weight details.
     */
    public MermaidExporter() {
        this(Direction.TOP_DOWN, false);
    }

    /**
     * Create an exporter with the given layout and detail level.
     *
     * @param detailed include constant weights when {@code true}
     */
    public MermaidExporter(Direction direction, boolean detailed) {
        this.direction = direction;
        this.detailed = detailed;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String serialize(Graph model) {
        CodeWriter writer = new CodeWriter();

        writer.writeLine("flowchart " + direction.id());
        writer.indent();

        Map<Node, String> ids = new HashMap<>();
        int counter = 0;

        for (Node in : model.input()) {
            ids.put(in, "n" + counter++);
        }

        for (Node node : model.topology()) {
            ids.put(node, "n" + counter++);
        }

        List<Node> inputNodes = model.input();

        for (int i = 0; i < inputNodes.size(); i++) {
            String uid = ids.get(inputNodes.get(i));

            writer.writeLine("%s[%s]".formatted(uid, "Input " + (i + 1)));
        }

        Map<String, String> weightLabels = new LinkedHashMap<>();
        Map<String, String> weightIds = new HashMap<>();
        int weightCounter = 0;

        if (detailed) {
            for (Node node : model.topology()) {
                if (node.layer() instanceof OnnxFormat.OnnxOperationLayer onnxLayer) {
                    String opType = OnnxIO.encodeType(onnxLayer.operation());
                    List<String> names = onnxLayer.inputNames();

                    for (int i = 0; i < names.size(); i++) {
                        String name = names.get(i);
                        Tensor constant = onnxLayer.constants().get(name);

                        if (constant != null && !weightLabels.containsKey(name)) {
                            weightLabels.put(name, "%s\\n%s".formatted(weightRole(opType, i), formatWeight(constant)));
                            weightIds.put(name, "w" + weightCounter++);
                        }
                    }
                }
            }

            for (Map.Entry<String, String> entry : weightLabels.entrySet()) {
                writer.writeLine("%s([%s])".formatted(weightIds.get(entry.getKey()), entry.getValue()));
            }
        }

        for (Node node : model.topology()) {
            String uid = ids.get(node);
            String name;

            if (node.layer() instanceof OnnxFormat.OnnxOperationLayer onnxLayer) {
                // ONNX operations share one layer type, so resolve the display name from the op type.
                String opType = OnnxIO.encodeType(onnxLayer.operation());

                name = opType != null ? opType : Commons.capitalize(node.name());
            } else {
                Activation activation = node.layer().activation();
                name = Commons.capitalize(node.name());

                if (activation != null) {
                    name += "\\n" + activation.name();
                }
            }

            writer.writeLine("%s[%s]".formatted(uid, name));
        }

        for (Node node : model.topology()) {
            String nodeUid = ids.get(node);

            for (Node input : node.inputs()) {
                String inputUid = ids.get(input);

                for (Shape inShape : input.outputShapes()) {
                    String label = format(inShape);

                    writer.writeLine("%s -->|\"%s\"| %s".formatted(inputUid, label, nodeUid));
                }
            }

            if (detailed && node.layer() instanceof OnnxFormat.OnnxOperationLayer onnxLayer) {
                for (String name : onnxLayer.inputNames()) {
                    if (onnxLayer.constants().containsKey(name)) {
                        writer.writeLine("%s -.-> %s".formatted(weightIds.get(name), nodeUid));
                    }
                }
            }
        }

        return writer.toString();
    }

    private String weightRole(String opType, int index) {
        if (opType == null) {
            return "weight";
        }

        return switch (opType) {
            case "Gemm" -> index == 2 ? "bias" : "weight";
            case "Add", "Sub" -> "bias";
            case "Mul", "Div" -> "scale";
            case "LayerNormalization" -> index == 2 ? "bias" : "scale";
            default -> "weight";
        };
    }

    private String formatWeight(Tensor tensor) {
        return Arrays.stream(tensor.shape())
            .mapToObj(String::valueOf)
            .collect(Collectors.joining("x"));
    }

    private String format(Shape shape) {
        String tmp = Arrays.stream(shape.dims())
            .mapToObj(String::valueOf)
            .collect(Collectors.joining("x"));

        return "(Bx%s)".formatted(tmp);
    }

    /**
     * Flowchart layout direction.
     * Use it to control the Mermaid rendering orientation.
     */
    public enum Direction {
        TOP_DOWN("TD"),
        DOWN_TOP("DT"),
        LEFT_RIGHT("LR"),
        RIGHT_LEFT("RL");

        private final String id;

        Direction(String id) {
            this.id = id;
        }

        /**
         * Return the Mermaid direction id.
         *
         * @return the direction id
         */
        public String id() {
            return id;
        }
    }
}
