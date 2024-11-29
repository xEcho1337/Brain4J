package net.echo.brain4j.training.data.nlp;

import net.echo.brain4j.nlp.LabelTransformer;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;

import java.util.ArrayList;
import java.util.List;

public class ConversationDataSet extends DataSet {
    private final LabelTransformer transformer;
    private final int maxLength;

    public ConversationDataSet(int maxLength, LabelTransformer transformer, String... conversations) {
        super(processConversations(conversations, maxLength, transformer));
        this.maxLength = maxLength;
        this.transformer = transformer;
    }

    private static DataRow[] processConversations(String[] conversations, int maxLength, LabelTransformer transformer) {
        List<DataRow> rows = new ArrayList<>();

        for (int i = 0; i < conversations.length - 1; i++) {
            double[] input = transformer.encode(conversations[i], maxLength);
            double[] output = transformer.encode(conversations[i + 1], maxLength);
            rows.add(new DataRow(input, output));
        }

        return rows.toArray(new DataRow[0]);
    }

    public void addConversation(String input, String output) {
        double[] encodedInput = transformer.encode(input, maxLength);
        double[] encodedOutput = transformer.encode(output, maxLength);
        getDataRows().add(new DataRow(encodedInput, encodedOutput));
    }
}

