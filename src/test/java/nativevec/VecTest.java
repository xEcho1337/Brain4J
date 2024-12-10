package nativevec;

import net.echo.brain4j.utils.NativeVector;
import net.echo.brain4j.utils.Vector;

public class VecTest {

    public static void main(String[] args) {
        int size = 50;

        double[] a = Vector.random(size).toArray();

        NativeVector nativeVector = new NativeVector(a);
        Vector vector = Vector.of(a);

        long nativeTook = 0;

        for (int i = 0; i < 100; i++) {
            nativeTook += evaluate(() -> nativeVector.sum2(a));
        }

        System.out.println("Native took " + (nativeTook / 1e6) + " ms");

        long javaTook = 0;

        for (int i = 0; i < 100; i++) {
            javaTook += evaluate(vector::sum);
        }

        System.out.println("Java took " + (javaTook / 1e6) + " ms");

        double improvement = (double) javaTook / nativeTook;

        System.out.println("Native improvement: " + improvement + "x");

    }

    private static long evaluate(Runnable runnable) {
        long start = System.nanoTime();
        runnable.run();

        return System.nanoTime() - start;
    }
}
