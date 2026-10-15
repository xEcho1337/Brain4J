package org.brain4j.core.importing.format;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import org.brain4j.math.commons.Commons;

/**
 * Maps string ids to classes and instance factories.
 * Use it to create importable objects from serialized ids.
 *
 * @param <T> the produced type
 * @param <A> the factory argument type
 */
public class GeneralRegistry<T, A> {

    private final Map<String, Class<? extends T>> idToClass = new HashMap<>();
    private final Map<Class<? extends T>, Set<String>> classToIds = new HashMap<>();
    private final Map<String, Function<A, T>> idToInstance = new HashMap<>();

    /**
     * Clear all registrations.
     */
    public void clear() {
        idToClass.clear();
        classToIds.clear();
        idToInstance.clear();
    }

    /**
     * Register a factory for the given id.
     */
    public void register(String identifier, Function<A, T> supplier) {
        idToInstance.put(identifier, supplier);
    }

    /**
     * Register a class for the given id.
     */
    public void register(String identifier, Class<? extends T> clazz) {
        idToClass.put(identifier, clazz);
        classToIds.computeIfAbsent(clazz, k -> new HashSet<>()).add(identifier);
    }

    /**
     * Find the primary id for the given class.
     *
     * @return the id, or {@code null} if missing
     */
    public String fromClass(Class<? extends T> clazz) {
        Set<String> ids = classToIds.get(clazz);

        return (ids == null || ids.isEmpty()) ? null : ids.iterator().next();
    }

    /**
     * Find all ids registered for the given class.
     *
     * @return the registered aliases, or an empty set
     */
    public Set<String> aliasesFromClass(Class<? extends T> clazz) {
        return classToIds.getOrDefault(clazz, Collections.emptySet());
    }

    /**
     * Find the class for the given id.
     *
     * @return the class, or {@code null} if missing
     */
    public Class<? extends T> fromId(String identifier) {
        return idToClass.get(identifier);
    }

    /**
     * Create an instance for the given id.
     *
     * @return the new instance
     */
    public T toInstance(String identifier) {
        return toInstance(identifier, null);
    }

    /**
     * Create an instance for the given id and argument.
     *
     * @param value the factory argument, may be {@code null}
     * @return the new instance
     */
    public T toInstance(String identifier, A value) {
        if (idToInstance.containsKey(identifier)) {
            return idToInstance.get(identifier).apply(value);
        }

        return Commons.newInstance(fromId(identifier));
    }
}
