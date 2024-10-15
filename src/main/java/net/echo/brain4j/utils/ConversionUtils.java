package net.echo.brain4j.utils;

/**
 * Utility class for conversions and value matching.
 */
public class ConversionUtils {

    /**
     * Finds the best matching enum constant based on output values.
     *
     * @param outputs array of output values
     * @param clazz   the enum class
     * @param <T>     the type of the enum
     * @return the best matching enum constant
     */
    public static <T extends Enum<T>> T findBestMatch(double[] outputs, Class<T> clazz) {
        return clazz.getEnumConstants()[indexOfMaxValue(outputs)];
    }

    /**
     * Finds the index of the maximum value in an array.
     *
     * @param inputs array of input values
     * @return index of the maximum value
     */
    public static int indexOfMaxValue(double[] inputs) {
        int index = 0;
        double max = inputs[0];

        for (int i = 1; i < inputs.length; i++) {
            if (inputs[i] > max) {
                max = inputs[i];
                index = i;
            }
        }

        return index;
    }
}