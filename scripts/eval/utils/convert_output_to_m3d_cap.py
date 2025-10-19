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

    m3d_cap_std = []
    for data, answer in tqdm(zip(data_input, answers_list), total=len(data_input)):
        outputs = data["outputs"].strip()
        if len(outputs) == 0:
            m3d_cap_std.append({
                "id": answer["id"],
                "outputs": outputs,
                "answers": answer["answer"],
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

        m3d_cap_std.append({
            "id": answer["id"],
            "outputs": outputs,
            "answers": answers,
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "meteor": meteor_score["meteor"],
            "bert_f1": sum(bert_score["f1"]) / len(bert_score["f1"]),
        })

    # calculate average scores
    bleu_scores = [x["bleu"] for x in m3d_cap_std]
    rouge1_scores = [x["rouge1"] for x in m3d_cap_std]
    meteor_scores = [x["meteor"] for x in m3d_cap_std]
    bert_f1_scores = [x["bert_f1"] for x in m3d_cap_std]
    print(f"BLEU: {sum(bleu_scores) / len(bleu_scores)}")
    print(f"ROUGE-1: {sum(rouge1_scores) / len(rouge1_scores)}")
    print(f"METEOR: {sum(meteor_scores) / len(meteor_scores)}")
    print(f"BertScore: {sum(bert_f1_scores) / len(bert_f1_scores)}")

    with open(args.output_path, "w") as f:
        json.dump(m3d_cap_std, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
