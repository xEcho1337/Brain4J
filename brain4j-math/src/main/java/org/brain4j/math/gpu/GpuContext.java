package org.brain4j.math.gpu;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.silicon.api.function.ComputeFunction;
import org.silicon.api.function.ComputeModule;
import org.silicon.api.kernel.ComputeQueue;

import org.brain4j.math.gpu.device.Device;

/**
 * Caches GPU kernels and modules per device. Use it to register compiled
 * functions once and reuse them across tensor operations.
 */
public class GpuContext {

    private static final Map<Device, Map<String, ComputeFunction>> KERNEL_CACHE = new ConcurrentHashMap<>();
    private static final Map<Device, Map<String, ComputeModule>> MODULE_CACHE = new ConcurrentHashMap<>();

    private GpuContext() {}

    /**
     * Register a compiled kernel for a device.
     *
     * @param device target device
     * @param kernelName kernel name
     * @param function compiled function
     */
    public static void register(Device device, String kernelName, ComputeFunction function) {
        KERNEL_CACHE.computeIfAbsent(device, d -> new ConcurrentHashMap<>())
            .put(kernelName, function);
    }

    /**
     * Extract a kernel from a module and register it.
     *
     * @param device target device
     * @param kernelName kernel name
     * @param module compiled module containing the kernel
     */
    public static void register(Device device, String kernelName, ComputeModule module) {
        try {
            ComputeFunction function = module.getFunction(kernelName);
            register(device, kernelName, function);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to register kernel " + kernelName, e);
        }

    }

    /**
     * Register several kernels from one module.
     *
     * @param device target device
     * @param module compiled module containing the kernels
     * @param kernelNames kernels to register
     */
    public static void registerAll(Device device, ComputeModule module, String... kernelNames) {
        for (String kernelName : kernelNames) {
            register(device, kernelName, module);
        }

    }

    /**
     * Store a compiled module for later lookup.
     *
     * @param device target device
     * @param moduleName module name
     * @param module compiled module
     */
    public static void storeModule(Device device, String moduleName, ComputeModule module) {
        MODULE_CACHE.computeIfAbsent(device, d -> new ConcurrentHashMap<>())
            .put(moduleName, module);
    }

    /**
     * Find a cached module.
     *
     * @param device target device
     * @param moduleName module name
     * @return cached module, or null if absent
     */
    public static ComputeModule getModule(Device device, String moduleName) {
        Map<String, ComputeModule> deviceModules = MODULE_CACHE.get(device);
        return deviceModules != null ? deviceModules.get(moduleName) : null;
    }

    /**
     * Find a registered kernel.
     *
     * @param device target device
     * @param kernelName kernel name
     * @return compiled function
     */
    public static ComputeFunction findFunction(Device device, String kernelName) {
        Map<String, ComputeFunction> deviceKernels = KERNEL_CACHE.get(device);

        if (deviceKernels == null) {
            throw new IllegalStateException("No kernels registered for device: " + device);
        }

        ComputeFunction function = deviceKernels.get(kernelName);

        if (function == null) {
            throw new IllegalStateException("Kernel " + kernelName + " not registered for device: " + device.name());
        }

        return function;
    }

    /**
     * Reuse the device queue or create a temporary one.
     *
     * @param device target device
     * @return handle closing only temporary queues
     */
    public static QueueHandle getOrCreateQueue(Device device) {
        ComputeQueue queue = device.queue();

        if (queue != null) {
            return new QueueHandle(queue, false);
        }

        return new QueueHandle(device.context().createQueue(), true);
    }

    /**
     * Drop all cached kernels and modules for a device.
     *
     * @param device target device
     */
    public static void clearCache(Device device) {
        KERNEL_CACHE.remove(device);
        MODULE_CACHE.remove(device);
    }

    /**
     * Drop every cached kernel and module.
     */
    public static void clearAllCaches() {
        KERNEL_CACHE.clear();
        MODULE_CACHE.clear();
    }

    /**
     * Wraps a queue and frees it only when temporary. Use it in
     * try-with-resources around kernel launches.
     *
     * @param queue queue to submit work to
     * @param temporary whether the queue must be freed on close
     */
    public record QueueHandle(ComputeQueue queue, boolean temporary) implements AutoCloseable {
        /**
         * {@inheritDoc}
         */
        @Override
        public void close() {
            if (temporary) {
                queue.await();
                queue.free();
            }

        }
    }
}
