## How to run evaluation of Streaming

1. Download the streaming video dataset from [OneDrive](https://drive.google.com/file/d/1VV020tllAvlvXyjjg7mJKmY8GPjVWfTz/view?usp=drive_link) and unzip it to `eval/example_data/frames`.

2. Set the `image_root` in `eval/example_data/input_full.jsonl` to the path of `eval/example_data/frames`.

3. To evaluate your own model, you should fomulate your output file in the same format as `eval/example_data/output_tmp.jsonl`.

4. Run the following command to evaluate the model:

```bash
python eval.py --input eval/example_data/output_tmp.jsonl --output eval/example_data/output_tmp_eval.json
```

5. The evaluation results will be saved to `eval/example_data/output_tmp_eval.json`.
