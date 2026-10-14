package org.brain4j.core.model;

import java.util.List;

import org.brain4j.core.layer.Layer;

/**
 * Contributes layers to a model definition.
 * Use it for reusable architecture fragments assembled into {@link ModelSpecs}.
 */
public interface ModelBlock {
    /**
     * Appends the layers of this block to the given list.
     */
    void appendTo(List<Layer> layers);
}
