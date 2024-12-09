package net.echo.brain4j.utils;

public class NativeVector {

    static {
        System.loadLibrary("nativevector");
    }

    private final long nativeHandle;

    public NativeVector(int size) {
        nativeHandle = init(size);
    }

    public NativeVector(double[] data) {
        nativeHandle = initWithData(data);
    }

    public native double length();

    public native double sum();

    public native void normalize();

    public native void scale(double value);

    public native double distance(NativeVector other);

    public native void free();

    private native long init(int size);

    private native long initWithData(double[] data);
}