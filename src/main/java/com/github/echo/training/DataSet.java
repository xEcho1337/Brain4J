package com.github.echo.training;

import java.util.ArrayList;
import java.util.List;

public class DataSet {

    private final List<DataRow> rows;

    public DataSet(DataRow... rows) {
        this.rows = new ArrayList<>(List.of(rows));
    }

    public DataSet add(DataRow row) {
        rows.add(row);
        return this;
    }

    public List<DataRow> rows() {
        return rows;
    }
}
