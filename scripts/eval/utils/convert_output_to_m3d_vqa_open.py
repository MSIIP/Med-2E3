import argparse
import json
from tqdm import tqdm

import evaluate


def main(args):
    with open(args.input_path, "r") as f:
        data_input = json.load(f)
    with open(args.answer_path, "r") as f:
        answers_list = json.load(f)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    m3d_vqa_open_std = []
    for data, answer in tqdm(zip(data_input, answers_list), total=len(data_input)):
        outputs = data["outputs"].strip()
        if len(outputs) == 0:
            m3d_vqa_open_std.append({
                "id": answer["id"],
                "outputs": outputs,
                "answers": answer["answer"],
                "question_type": answer["question_type"],
                "bleu": 0,
                "rouge1": 0,
                "meteor": 0,
                "bert_f1": 0,
            })
            continue

        answers = answer["answer"].strip()
        bleu_score = bleu.compute(predictions=[outputs], references=[answers], max_order=1)
        rouge_score = rouge.compute(predictions=[outputs], references=[answers], rouge_types=["rouge1"])
        meteor_score = meteor.compute(predictions=[outputs], references=[answers])
        bert_score = bertscore.compute(predictions=[outputs], references=[answers], lang="en")

        m3d_vqa_open_std.append({
            "id": answer["id"],
            "outputs": outputs,
            "answers": answers,
            "question_type": answer["question_type"],
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "meteor": meteor_score["meteor"],
            "bert_f1": sum(bert_score["f1"]) / len(bert_score["f1"]),
        })

    metrics = ["bleu", "rouge1", "meteor", "bert_f1"]
    for metric in metrics:
        metric_list = []
        qtypes = sorted(set([r["question_type"] for r in m3d_vqa_open_std]))
        for qtype in qtypes:
            scores = [r[metric] for r in m3d_vqa_open_std if r["question_type"] == qtype]
            avg_score = sum(scores) / len(scores)
            metric_list.append(avg_score)
            print(f"Question type: {qtype}, Average {metric}: {avg_score}")
        print(f"Average {metric}: {sum(metric_list) / len(metric_list)}")

        scores = [r[metric] for r in m3d_vqa_open_std]
        avg_score = sum(scores) / len(scores)
        print(f"{metric}: {avg_score}\n")

    with open(args.output_path, "w") as f:
        json.dump(m3d_vqa_open_std, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
