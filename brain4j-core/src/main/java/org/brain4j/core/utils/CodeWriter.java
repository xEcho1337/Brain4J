package org.brain4j.core.utils;

/**
 * Builds indented text output line by line.
 * Use it to serialize models to readable formats.
 */
public class CodeWriter {

    private int indentation = 0;
    private String buffer;

    /**
     * Create a writer with empty content.
     */
    public CodeWriter() {
        this.buffer = "";
    }

    /**
     * Increase the indentation level.
     */
    public void indent() {
        indentation++;
    }

    /**
     * Decrease the indentation level.
     */
    public void unindent() {
        indentation--;
    }

    /**
     * Append indented text without a newline.
     */
    public void write(String text) {
        buffer += " ".repeat(indentation * 4) + text;
    }

    /**
     * Append an indented line with a newline.
     */
    public void writeLine(String line) {
        write(line + "\n");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return buffer;
    }
}
