package nativevec;

import net.echo.brain4j.utils.NativeVector;
import net.echo.brain4j.utils.Vector;

public class VectorPerformance {

    public static void main(String[] args) {
        int size = 500;
        double[] data = new double[size];

        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }

        NativeVector nativeVector = new NativeVector(data);
        Vector vector = Vector.of(data);

        long start = System.nanoTime();
        nativeVector.sum2(data);
        long took = System.nanoTime() - start;

        System.out.println("Took " + took + " ns");

        start = System.nanoTime();
        vector.sum();
        took = System.nanoTime() - start;

        System.out.println("Took " + took + " ns");
    }
}
