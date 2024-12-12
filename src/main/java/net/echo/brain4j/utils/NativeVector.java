package net.echo.brain4j.utils;

public class NativeVector {

    static {
        System.load("C:\\Users\\xrefl\\Projects\\Java\\Brain4J\\src\\test\\java\\nativevec\\natives\\brain4j_backend.dll");
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