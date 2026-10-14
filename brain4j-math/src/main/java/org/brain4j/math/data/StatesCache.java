package org.brain4j.math.data;

import java.util.HashMap;
import java.util.Map;

import org.brain4j.math.tensor.Tensor;

/**
 * Caches intermediate tensors and layer states for one forward pass.
 * Use it to share activations between forward and backward passes.
 */
public class StatesCache {

    private final Map<Object, Tensor> tensorCache;
    private final Map<String, Tensor[]> states;
    private final boolean training;

    /**
     * Creates a cache for a training pass.
     *
     * @return a training-enabled cache
     */
    public static StatesCache withTraining() {
        return new StatesCache(true);
    }

    /**
     * Creates an inference cache.
     */
    public StatesCache() {
        this(false);
    }

    /**
     * Creates a cache for training or inference.
     */
    public StatesCache(boolean training) {
        this.training = training;
        this.states = new HashMap<>();
        this.tensorCache = new HashMap<>();
    }

    /**
     * Checks whether this cache backs a training pass.
     *
     * @return true for training, false for inference
     */
    public boolean isTraining() {
        return training;
    }

    /**
     * Returns the cached tensor for the given key.
     *
     * @return the cached tensor, or {@code null} if absent
     */
    public Tensor get(Object key) {
        return tensorCache.get(key);
    }

    /**
     * Stores a tensor under the given key.
     */
    public void set(Object key, Tensor value) {
        tensorCache.put(key, value);
    }

    /**
     * Returns the stored states for the given key and id.
     *
     * @return the stored tensors, or {@code null} if absent
     */
    public Tensor[] getStates(Object key, String id) {
        return states.get(key.hashCode() + id);
    }

    /**
     * Stores states for the given key and id.
     */
    public void setStates(Object key, String id, Tensor... values) {
        states.put(key.hashCode() + id, values);
    }
}
