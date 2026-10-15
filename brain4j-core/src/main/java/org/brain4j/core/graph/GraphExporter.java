package org.brain4j.core.graph;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

import org.brain4j.core.model.impl.Graph;

/**
 * Converts a graph model to a text representation.
 * Use it to export architectures for visualization or debugging.
 */
public interface GraphExporter {
    /**
     * Serialize the model to its text representation.
     *
     * @return the serialized model
     */
    String serialize(Graph model);

    /**
     * Write the serialized model to the given file.
     *
     * @param outFile the destination file
     */
    default void export(Graph model, File outFile) {
        try (FileOutputStream fileOutputStream = new FileOutputStream(outFile)) {
            try (BufferedOutputStream outputStream = new BufferedOutputStream(fileOutputStream)) {
                String out = serialize(model);

                outputStream.write(out.getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
