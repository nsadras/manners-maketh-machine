# Manners Maketh Machine

An investigation into the effect of prompt tone on LLM accuracy. 

This experiment prepends various tonally-charged phrases, ranging from polite to threatening, to selected questions from the [CAIS MMLU dataset](https://huggingface.co/datasets/cais/mmlu), to test whether tone impacts task performance.

## Usage

1) install local models:

    ```bash
    ollama pull gemma4:12b
    ollama pull gemma4:e2b
    ollama pull qwen3.5:9b
    ```

2) run experiment:

    ```bash
    uv run main.py
    ```
    writes results to `results_<datetime>.csv`

3) plot results:

    ```bash
    uv run plot_results.py results_<datetime>.csv
    ```
