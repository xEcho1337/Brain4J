package org.brain4j.core.importing.io;

import java.io.File;

import org.brain4j.core.importing.format.BinaryFormat;
import org.brain4j.core.model.Model;

/**
 * Saves and loads models through a binary format adapter.
 * Use it to persist trained models to disk.
 */
public class ModelIO {

    /**
     * Serializes the given model to the given path.
     */
    public static <T extends Model> void save(T model, BinaryFormat<T> adapter, File path) {
        adapter.serialize(model, path);
    }

    /**
     * Deserializes a model from the given path.
     *
     * @return the loaded model
     */
    public static <T extends Model> T load(BinaryFormat<T> adapter, File path) {
        return adapter.deserialize(path);
    }
}
