package net.echo.brain4j.nlp;

public class AlphabetInitialization {

    public static AlphabetInitialization NORMAL = new AlphabetInitialization(" abcdefghijklmnopqrstuvwxyz0123456789@#!$%^&*()-_=+[]{}|;:'\",.<>?/`~\\");
    public static AlphabetInitialization SEMANTIC = new AlphabetInitialization(" a4b8cde3fg9hi1jklmno0pqrs5t7uvwxyz26@#!$%^&*()-_=+[]{}|;:'\",.<>?/`~\\");

    private final String alphabet;

    AlphabetInitialization(String alphabet) {
        this.alphabet = alphabet;
    }

    public static AlphabetInitialization create(String alphabet) {
        return new AlphabetInitialization(alphabet);
    }

    public String getAlphabet() {
        return alphabet;
    }
}
