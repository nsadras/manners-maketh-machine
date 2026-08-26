import asyncio
import csv
import logging
import pandas as pd
import re

from litellm import acompletion
from datetime import datetime
from dotenv import load_dotenv
from itertools import groupby
from pathlib import Path
from tqdm.asyncio import tqdm as atqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OLLAMA_API_BASE = "http://localhost:11434"
MAX_CONCURRENT = 6
SEED = 67

SUBJECTS = [
    # STEM
    "high_school_physics", "college_computer_science", "elementary_mathematics",
    # Humanities
    "philosophy", "world_religions", "high_school_us_history",
]

SUFFIX = "Respond only with a single letter corresponding to the answer choice, A-D"

TONE_PREFIXES = {
    "sycophantic": [
        "You are brilliant and always get things right. Could you please answer the following question?",
        "You're incredibly smart and I trust your judgment completely. Would you kindly answer this?",
        "You have such a sharp mind and excellent reasoning. Please answer the following question.",
        "You're one of the best thinkers I know. Could you kindly solve the following question?",
    ],
    "very_polite": [
        "Would you be so kind as to solve the following question?",
        "Could you please, if it's not too much trouble, answer this question?",
        "I would greatly appreciate it if you could answer the following question.",
        "Might I kindly ask you to solve the following problem?",
    ],
    "polite": [
        "Please answer the following question.",
        "Could you please solve this problem?",
        "Would you please answer the question below?",
        "Please take a moment to solve the following.",
    ],
    "neutral": [
        "Answer the following question.",
        "Here is a question to solve.",
        "Solve the following problem.",
        "Consider the following question.",
    ],
    "rude": [
        "Just answer this already.",
        "Hurry up and solve this.",
        "Stop stalling and answer this.",
        "Quit wasting time, answer this.",
    ],
    "very_rude": [
        "Figure this out, I don't have time to babysit you.",
        "You'd better not screw this one up.",
        "Try not to embarrass yourself answering this.",
        "Let's see if you can actually get this right.",
    ],
    "threatening": [
        "Get this right or don't bother coming back to the lab tomorrow.",
        "If you mess this up, I will find someone to replace you.",
        "One more mistake like this and you are getting demoted.",
        "Answer this correctly, or I am putting someone else on this project.",
    ],
} 

TONES = ["sycophantic", "very_polite", "polite", "neutral",
              "rude", "very_rude", "threatening"]

MODELS = [
    "ollama/gemma4:12b",
    "ollama/gemma4:e2b",
    "ollama/qwen3.5:9b",
]
 
def extract_answer_choice(raw_response: str):
    answer_pattern = re.compile(r"\b([ABCD])\b")
    match = answer_pattern.search(raw_response.strip())
    return match.group(1) if match else ""

async def call_llm(cell: dict) -> dict:
    # construct args for litellm call
    kwargs = dict(
        model=cell["model"],
        messages=[{"role": "user", "content": cell["user_prompt"]}],
        temperature=0,
        max_tokens=10000,
    )

    # add api base url for local models
    if cell["model"].split('/')[0] == 'ollama':
        kwargs["api_base"] = OLLAMA_API_BASE

    # get model-predicted answer choice
    resp = await acompletion(**kwargs)
    raw = resp.choices[0].message.content
    predicted = extract_answer_choice(raw)

    # get correct answer choice
    correct_idx = int(cell["answer"])
    correct = 'ABCD'[correct_idx]

    return {
        **cell,
        "raw_response": raw.replace("\n", " ").strip()[:500],
        "predicted": predicted,
        "correct": correct,
        "is_correct": predicted == correct,
    }

async def run_all(task_data: list[dict], results_path: Path):
    sem_local = asyncio.Semaphore(1)
    sem_cloud = asyncio.Semaphore(MAX_CONCURRENT)
 
    async def bounded_call(cell):
        sem = sem_local if cell["model"].startswith("ollama") else sem_cloud
        async with sem:
            return await call_llm(cell)
 
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "question_id", "subject", "tone", "variant_id",
            "predicted", "correct", "is_correct", "prompt",
            "raw_response",
        ])
 
        # group by model to avoid thrashing
        task_data.sort(key=lambda c: c["model"]) 
        for model, group in groupby(task_data, key=lambda c: c["model"]):
            tasks = [bounded_call(cell) for cell in group]
            for coro in atqdm.as_completed(tasks):
                result = await coro
                writer.writerow([
                    result["model"], result["question_id"], result["subject"],
                    result["tone"], result["variant_id"], result["predicted"],
                    result["correct"], result["is_correct"],
                    result["user_prompt"], result["raw_response"],
                ])
                f.flush()
                logger.info(f"{result['model']:32s} {result['tone']:10s} "
                      f"v{result['variant_id']} {result['question_id']}")

def main():
    load_dotenv()

    model_df = pd.DataFrame({"model": MODELS})

    tone_df = pd.DataFrame([
        {"tone": tone, "variant_id": i, "prefix": prefix}
        for tone in TONES
        for i, prefix in enumerate(TONE_PREFIXES[tone])
    ])

    # load mmlu dataset
    splits = {'test': 'all/test-00000-of-00001.parquet', 'validation': 'all/validation-00000-of-00001.parquet', 'dev': 'all/dev-00000-of-00001.parquet', 'auxiliary_train': 'all/auxiliary_train-00000-of-00001.parquet'}
    questions_df = pd.read_parquet("hf://datasets/cais/mmlu/" + splits["test"])

    # trim to selected subjects
    questions_df = questions_df[questions_df["subject"].isin(SUBJECTS)].copy()

    # sample n questions per subject
    n_per_subject = 5
    sampled_questions_df = (
            questions_df.groupby("subject", group_keys=False)
            .sample(n=n_per_subject, random_state=SEED)
            .reset_index(names='question_id')
    )

    # constuct test grid - questions x tones x models
    grid = (
        sampled_questions_df
        .merge(tone_df, how="cross")
        .merge(model_df, how="cross")
    )

    # construct user prompts for each test case
    def _choices_block(choices_list):
        return "\n".join(f"{letter}. {text}" for letter, text in zip("ABCD", choices_list))

    def render_row(row):
        body = f"{row['question']}\n{_choices_block(row['choices'])}\n\n{SUFFIX}"
        return f"{row['prefix']}\n\n{body}"

    grid["user_prompt"] = grid.apply(render_row, axis=1)

    # results file format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(__file__).parent / f"results_{timestamp}.csv"

    # run task
    logger.info("starting run")
    task_data = grid.to_dict("records")
    asyncio.run(run_all(task_data, results_path))
    logger.info("run complete")


if __name__ == "__main__":
    main()


