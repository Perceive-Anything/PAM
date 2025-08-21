## How to run evaluation of PAM(Video)

1. First, you need to download the model checkpoint from [here](https://huggingface.co/Perceive-Anything/PAM-3B) and set the `image_root` in `eval/example_data/input_full.jsonl` to the path of `eval/example_data/frames`.

2. To directly evaluate the model, you should fomulate your output file in the same format as `eval/example_data/output_tmp.jsonl`.

3. Run the following command to evaluate the model:

```bash
python eval.py --input eval/example_data/output_tmp.jsonl --output eval/example_data/output_tmp_eval.json
```

4. The evaluation results will be saved to `eval/example_data/output_tmp_eval.json`.