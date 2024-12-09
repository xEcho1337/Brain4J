package nativevec;

import net.echo.brain4j.utils.HelloWorldJNI;

public class HelloWorld {

    public static void main(String[] args) {
        HelloWorldJNI world = new HelloWorldJNI();
        world.sayHello();
    }
}
