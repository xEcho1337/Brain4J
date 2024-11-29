package net.echo.brain4j.nlp;

import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LabelTransformer {

    private final Map<Character, Integer> charToIndexMap = new HashMap<>();
    private final Map<Integer, Character> indexToCharMap = new HashMap<>();

    public LabelTransformer(AlphabetInitialization technique) {
        char[] characters = technique.getAlphabet().toCharArray();

        for (int i = 0; i < characters.length; i++) {
            charToIndexMap.put(characters[i], i + 1);
            indexToCharMap.put(i + 1, characters[i]);
        }
    }

    public char transform(int index) {
        return indexToCharMap.get(index);
    }

    public DataSet getDataSet(Map<String, Integer> labels, int length) {
        List<DataRow> rows = new ArrayList<>();

        for (Map.Entry<String, Integer> entry : labels.entrySet()) {
            rows.add(new DataRow(encode(entry.getKey(), length), entry.getValue()));
        }

        return new DataSet(rows.toArray(new DataRow[0]));
    }

    public double[] encode(String text, int length) {
        double[] encoded = new double[length];

        text = text.length() > length ? text.substring(0, length).toLowerCase() : text.toLowerCase();

        for (int i = 0; i < text.length(); i++) {
            char currentChar = text.charAt(i);
            encoded[i] = charToIndexMap.getOrDefault(currentChar, 0);
        }

        return encoded;
    }

    public String decode(double[] encoded) {
        StringBuilder decoded = new StringBuilder();

        for (double v : encoded) {
            decoded.append(indexToCharMap.get((int) v));
        }

        return decoded.toString();
    }
}
