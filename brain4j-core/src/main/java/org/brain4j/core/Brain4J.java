package org.brain4j.core;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Entry point for framework setup and device discovery.
 * Use it to configure logging and access GPU devices.
 *
 * @author xEcho1337
 * @author Adversing
 */
public class Brain4J {

    private static boolean logging = true;
    private static boolean disableColors = false;
    private static boolean fixedOutStream = false;
    private static int decimalDigits = 4;

    /**
     * Returns the current framework version.
     *
     * @return the version string
     */
    public static String getVersion() {
        return "3.0";
    }

    /**
     * Forces UTF-8 console output once.
     */
    public static void fixConsole() {
        if (fixedOutStream) {
            return;
        }

        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        fixedOutStream = true;
    }

    /**
     * Checks whether training progress is logged to the console.
     *
     * @return {@code true} if logging is enabled
     */
    public static boolean isLogging() {
        fixConsole();
        return logging;
    }

    /**
     * Enables or disables training progress logging.
     */
    public static void setLogging(boolean logging) {
        Brain4J.logging = logging;
    }

    /**
     * Checks whether ANSI colors are disabled.
     *
     * @return {@code true} if colors are disabled
     */
    public static boolean isDisableColors() {
        return disableColors;
    }

    /**
     * Enables or disables ANSI colors.
     */
    public static void setDisableColors(boolean disableColors) {
        Brain4J.disableColors = disableColors;
    }

    /**
     * Returns the decimal precision used in logs.
     *
     * @return the displayed decimal digits
     */
    public static int getDecimalDigits() {
        return decimalDigits;
    }

    /**
     * Replaces the decimal precision used in logs.
     */
    public static void setDecimalDigits(int decimalDigits) {
        Brain4J.decimalDigits = decimalDigits;
    }

    /**
     * Returns names of all detected GPU devices.
     *
     * @return a comma-separated device list, or empty if none
     */
    public static String getAvailableDevices() {
        return String.join(", ", DeviceUtils.allDeviceNames());
    }

    /**
     * Compiles GPU kernels on the given device.
     */
    public static void initKernels(Device device) {
        GpuTensor.initKernels(device);
    }

    /**
     * Returns the first available GPU device.
     *
     * @return the first device, or an empty optional if none is available
     */
    public static Optional<Device> firstDevice() {
        try {
            List<String> devices = DeviceUtils.allDeviceNames();

            if (devices.isEmpty()) {
                return Optional.empty();
            }

            Device device = DeviceUtils.findDevice(devices.getFirst());

            if (device == null) {
                return Optional.empty();
            }

            Brain4J.initKernels(device);
            return Optional.of(device);
        } catch (IllegalStateException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns all available GPU devices.
     *
     * @return the detected devices
     */
    public static List<Device> getAllDevices() {
        List<Device> devices = new ArrayList<>();

        for (String device : DeviceUtils.allDeviceNames()) {
            devices.add(DeviceUtils.findDevice(device));
        }

        return devices;
    }

    /**
     * Finds a GPU device by name.
     *
     * @return the matching device, or {@code null} if not found
     */
    public static Device findDevice(String deviceName) {
        return DeviceUtils.findDevice(deviceName);
    }
}
