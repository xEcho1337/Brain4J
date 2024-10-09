package net.echo.brain4j.utils;

public class ConversionUtils {

    public static <T extends Enum<T>> Enum<T> findBestMatch(double[] outputs, Class<T> clazz) {
        Enum<T>[] constants = clazz.getEnumConstants();

        return constants[findMaxValuePosition(outputs)];
    }

    public static int findMaxValuePosition(double[] inputs) {
        int index = 0;
        double max = 0.0;

        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] > max) {
                max = inputs[i];
                index = i;
            }
        }

        return index;
    }
}
