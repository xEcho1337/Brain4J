package net.echo.brain4j.training.data.nlp;

import net.echo.brain4j.nlp.LabelTransformer;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.Arrays;
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
            List<Vector> input = transformer.textToEmbedding(conversations[i], maxLength);
            List<Vector> output = transformer.textToEmbedding(conversations[i + 1], maxLength);

            for (int j = 0; j < input.size(); j++) {
                Vector inputVec = input.get(i);

                if (j < output.size()) {
                    Vector outputVec = output.get(i);

                    rows.add(new DataRow(inputVec, outputVec));
                }
            }
        }

        return rows.toArray(new DataRow[0]);
    }

    public void addConversation(String input, String output) {
        double[] encodedInput = transformer.encode(input, maxLength);
        double[] encodedOutput = transformer.encode(output, maxLength);
        getDataRows().add(new DataRow(new Vector(encodedInput), new Vector(encodedOutput)));
    }
}

