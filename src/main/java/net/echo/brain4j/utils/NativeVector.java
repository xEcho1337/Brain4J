package net.echo.brain4j.utils;

public class NativeVector {

    static {
        System.load("/Users/echo/IdeaProjects/Brain4J/src/test/java/nativevec/natives/libbrain4j_backend.dylib");
    }

    private final long nativeHandle;

    public NativeVector(int size) {
        nativeHandle = init(size);
    }

    public NativeVector(double[] data) {
        nativeHandle = initWithData(data, data.length);
    }

    public long getNativeHandle() {
        return nativeHandle;
    }

    public native double[] convolute(double[] a, double[] b);

    public native double sum2(double[] input);

    private native long init(int size);

    private native long initWithData(double[] data, int length);
}