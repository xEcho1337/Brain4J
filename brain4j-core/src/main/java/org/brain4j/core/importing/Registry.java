package org.brain4j.core.importing;

import java.util.HashMap;
import java.util.Map;

import org.brain4j.core.codec.Codec;

/**
 * Maps serializable types to codecs by id and class.
 * Use it to look up the codec needed for import or export.
 *
 * @param <T> the base handled type
 */
public class Registry<T> {

    private final Map<String, Codec<? extends T>> byType = new HashMap<>();
    private final Map<Class<?>, Codec<? extends T>> byClass = new HashMap<>();

    /**
     * Register the given codecs for lookup.
     *
     * @param codecs the codecs to store
     */
    @SafeVarargs
    public final void put(Codec<? extends T>... codecs) {
        for (Codec<? extends T> codec : codecs) {
            byType.put(codec.type(), codec);
            byClass.put(codec.targetClass(), codec);
        }
    }

    /**
     * Find the codec for the given class.
     *
     * @return the matching codec, or {@code null} if missing
     */
    @SuppressWarnings("unchecked")
    public <S extends T> Codec<T> get(Class<S> clazz) {
        return (Codec<T>) byClass.get(clazz);
    }

    /**
     * Find the codec for the given type id.
     *
     * @return the matching codec, or {@code null} if missing
     */
    @SuppressWarnings("unchecked")
    public Codec<T> get(String type) {
        return (Codec<T>) byType.get(type);
    }
}
