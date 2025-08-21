## How to run evaluation of Streaming

1. Set the `image_root` in `eval/example_data/input_full.jsonl` to the path of `eval/example_data/frames`.

2. To evaluate your own model, you should fomulate your output file in the same format as `eval/example_data/output_tmp.jsonl`.

3. Run the following command to evaluate the model:

```bash
python eval.py --input eval/example_data/output_tmp.jsonl --output eval/example_data/output_tmp_eval.json
```

4. The evaluation results will be saved to `eval/example_data/output_tmp_eval.json`.
