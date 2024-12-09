package net.echo.brain4j.utils;

public class HelloWorldJNI {

    static {
        System.loadLibrary("nativelib");
    }

    public native void sayHello();
}