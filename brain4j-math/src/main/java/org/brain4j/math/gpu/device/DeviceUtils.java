package org.brain4j.math.gpu.device;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;

import org.silicon.api.Silicon;
import org.silicon.api.backend.BackendType;
import org.silicon.api.backend.ComputeBackend;
import org.silicon.api.device.ComputeDevice;

import org.brain4j.math.commons.Commons;

/**
 * Finds GPU devices and loads kernel sources. Use it to select a backend
 * before creating tensors on the GPU.
 */
public class DeviceUtils {

    private DeviceUtils() {}

    /**
     * Find the default GPU device.
     *
     * @return default device, or null if unavailable
     */
    public static Device findDevice() {
        return findDevice(0);
    }

    /**
     * Find a GPU device by index.
     *
     * @param index device index
     * @return device, or null if unavailable
     */
    public static Device findDevice(int index) {
        try {
            return new Device(index);
        } catch (Exception e) {
            return null;
        }

    }

    /**
     * Find a GPU device by name fragment.
     *
     * @param name case-insensitive name fragment, or null for the default device
     * @return matching device, or null if absent
     */
    public static Device findDevice(String name) {
        if (name == null) {
            return findDevice(0);
        }

        for (int i = 0; i < 8; i++) {
            Device device = null;

            try {
                device = new Device(i);

                if (device.name().toLowerCase().contains(name.toLowerCase())) {
                    return device;
                }

                freeQuietly(device);
            } catch (Exception e) {
                freeQuietly(device);
                break;
            }

        }

        return null;
    }

    /**
     * List all visible device names.
     *
     * @return device names in index order
     */
    public static List<String> allDeviceNames() {
        ComputeBackend backend = Silicon.backend();

        if (backend == null) {
            throw new IllegalStateException("Backend is null! Make sure to import at least one backend!");
        }

        List<String> names = new ArrayList<>();

        int count = backend.deviceCount();

        for (int i = 0; i < count; i++) {
            try {
                ComputeDevice device = Silicon.createDevice(i);
                names.add(device.name());
            } catch (Throwable e) {
                break;
            }

        }

        return names;
    }

    /**
     * Get the active backend type.
     *
     * @return active backend type
     */
    public static BackendType getBackendType() {
        return Silicon.backend().type();
    }

    /**
     * Get the active compute backend.
     *
     * @return active backend
     */
    public static ComputeBackend getBackend() {
        return Silicon.backend();
    }

    /**
     * Select the compute backend.
     *
     * @param backendType backend to activate
     */
    public static void chooseBackend(BackendType backendType) {
        Silicon.chooseBackend(backendType);
    }

    /**
     * Read a kernel source from the classpath.
     *
     * @param resourcePath classpath resource path
     * @return file contents as UTF-8 text
     */
    public static String readKernelSource(String resourcePath) {
        try (InputStream input = DeviceUtils.class.getResourceAsStream(resourcePath)) {
            if (input == null) {
                throw Commons.illegalArgument("Resource not found: %s", resourcePath);
            }

            return new String(input.readAllBytes(), UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read kernel source from: " + resourcePath, e);
        }

    }

    /**
     * Check whether the vector incubator module is present.
     *
     * @return true if SIMD execution is available
     */
    public static boolean isSimdAvailable() {
        return ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    }

    /**
     * Translate a backend error code to text.
     *
     * @param code backend status code
     * @return human-readable status
     */
    public static String getErrorCode(int code) {
        return code == 0 ? "SUCCESS" : "Silicon backend error code " + code;
    }

    /**
     * Fail when a backend call returns an error.
     *
     * @param profiler operation name used in the message
     * @param err backend status code, zero means success
     */
    public static void checkError(String profiler, int err) {
        if (err == 0) {
            return;
        }

        throw new RuntimeException("GPU(" + profiler + ") - " + getErrorCode(err));
    }

    private static UnsupportedOperationException unsupportedRawGpuHandle() {
        return new UnsupportedOperationException(
            "Raw native device handles are not available in the Silicon GPU backend"
        );
    }

    private static void freeQuietly(Device device) {
        if (device == null) {
            return;
        }

        try {
            device.free();
        } catch (RuntimeException ignored) {
        }

    }
}
