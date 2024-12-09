package nativevec;

import net.echo.brain4j.utils.NativeVector;
import net.echo.brain4j.utils.Vector;

public class VecTest {

    public static void main(String[] args) {
        /*System.setProperty("java.library.path", "./natives");
        System.setProperty("sun.library.path", "./natives");*/


        System.out.println(System.getProperty("java.library.path"));
        int size = 50;
        double[] data = new double[size];

        for (int i = 0; i < size; i++) {
            data[i] = Math.random();
        }

        NativeVector nativeVector = new NativeVector(data);
        Vector vector = Vector.of(data);

        long start = System.nanoTime();
        nativeVector.normalize();
        long took = System.nanoTime() - start;

        System.out.println("Took " + took + " ns");

        start = System.nanoTime();
        vector.normalize();
        took = System.nanoTime() - start;

        System.out.println("Took " + took + " ns");
    }
}
