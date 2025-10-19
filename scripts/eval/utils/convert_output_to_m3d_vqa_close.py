import argparse
import json
from tqdm import tqdm


def main(args):
    with open(args.input_path, "r") as f:
        data_input = json.load(f)
    with open(args.answer_path, "r") as f:
        answers_list = json.load(f)

    m3d_vqa_close_std = []
    for data, answer in tqdm(zip(data_input, answers_list), total=len(data_input)):
        outputs = data["outputs"].strip()
        correct = (answer["answer_choice"] + ".") in outputs

        m3d_vqa_close_std.append({
            "id": answer["id"],
            "outputs": outputs,
            "answers": answer["answer"].strip(),
            "question_type": answer["question_type"],
            "correct": correct,
        })

    metrics = ["correct"]
    for metric in metrics:
        metric_list = []
        qtypes = sorted(set([r["question_type"] for r in m3d_vqa_close_std]))
        for qtype in qtypes:
            scores = [r[metric] for r in m3d_vqa_close_std if r["question_type"] == qtype]
            avg_score = sum(scores) / len(scores)
            metric_list.append(avg_score)
            print(f"Question type: {qtype}, Average {metric}: {avg_score}")
        print(f"Average {metric}: {sum(metric_list) / len(metric_list)}")

        scores = [r[metric] for r in m3d_vqa_close_std]
        avg_score = sum(scores) / len(scores)
        print(f"{metric}: {avg_score}\n")

    with open(args.output_path, "w") as f:
        json.dump(m3d_vqa_close_std, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
