package org.brain4j.core.importing.format;

import java.io.File;

/**
 * Reads and writes a model to a binary file.
 * Use it to support pluggable model file formats.
 *
 * @param <T> the model type
 */
public interface BinaryFormat<T> {
    /**
     * Load a model from the given file.
     *
     * @param file the source file
     * @return the deserialized model
     */
    T deserialize(File file);

    /**
     * Save the model to the given file.
     *
     * @param file the destination file
     */
    void serialize(T input, File file);
}
