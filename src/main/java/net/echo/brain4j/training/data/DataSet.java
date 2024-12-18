package net.echo.brain4j.training.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DataSet {

    private final List<DataRow> dataRows;
    private final List<List<DataRow>> partitions = new ArrayList<>();

    public DataSet(List<DataRow> dataRows) {
        this.dataRows = dataRows;
    }

    public DataSet(DataRow... rows) {
        this.dataRows = new ArrayList<>(Arrays.asList(rows));
    }

    public List<DataRow> getDataRows() {
        return dataRows;
    }

    public boolean isPartitioned() {
        return !partitions.isEmpty();
    }

    public List<List<DataRow>> getPartitions() {
        return partitions;
    }

    private List<DataRow> divide(List<DataRow> rows, double batches, int offset) {
        int start = (int) Math.min(offset * batches, rows.size());
        int stop = (int) Math.min((offset + 1) * batches, rows.size());

        return rows.subList(start, stop);
    }

    public void partition(int batches) {
        this.partitions.clear();

        int rowsPerBatch = dataRows.size() / batches;

        for (int i = 0; i < batches; i++) {
            this.partitions.add(divide(dataRows, rowsPerBatch, i));
        }
    }

    /**
     * Randomly shuffles the dataset, making the training more efficient.
     */
    public void shuffle() {
        Collections.shuffle(dataRows);
    }
}
