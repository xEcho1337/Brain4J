package org.brain4j.core.utils;

import java.util.Map;

import org.brain4j.core.Brain4J;

import static org.brain4j.math.Constants.GRAY;
import static org.brain4j.math.Constants.LIGHT_BLUE;
import static org.brain4j.math.Constants.LIGHT_GREEN;
import static org.brain4j.math.Constants.LIGHT_YELLOW;
import static org.brain4j.math.Constants.MAGENTA;
import static org.brain4j.math.Constants.RESET;
import static org.brain4j.math.Constants.WHITE;

/**
 * Formats strings with inline color tags.
 * Use it to print readable training logs and progress.
 */
public class Colored {

    private static final Map<String, String> COLORS = Map.of(
        "yellow", LIGHT_YELLOW,
        "white", WHITE,
        "blue", LIGHT_BLUE,
        "green", LIGHT_GREEN,
        "gray", GRAY,
        "magenta", MAGENTA,
        "reset", RESET
    );

    /**
     * Render a template with format args and color tags.
     *
     * @param template the format string with optional color tags
     * @param args arguments referenced by the format specifiers
     * @return the rendered string with ANSI codes applied
     */
    public static String renderText(String template, Object... args) {
        String formatted = String.format(template, args);

        for (var entry : COLORS.entrySet()) {
            String value = Brain4J.isDisableColors() ? "" : entry.getValue();
            formatted = formatted.replace("<" + entry.getKey() + ">", value);
        }

        return formatted;
    }
}
