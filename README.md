Variance predictions of atmospheric variables

# Data Prep

## Prior to running data prep

Copy, edit and rename `data_prep/python/script_variables_template.py` to `data_prep/python/script_variables.py` such that the variables satisfy the needs of your setup. This must include root directories for reading and storing the data.

## Running data prep

Ensure python3 is ready and running the following command:

```
python loop_files.py <day>
```

where `<day>` is an integer. In this case, day takes values 1, 2, ..., 10.
`run_loop_files.sh` is already setup for run this.