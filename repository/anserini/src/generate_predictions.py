import sys
import argparse
import time
import json


def load_jsonl(input_path: str) -> list:
    """Read list of objects from a JSON lines file."""
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data

def load_anserini_predictions(path: str) -> dict:
    with open(path) as fr:
        predictions = {}
        for line in fr.readlines():
            idx, pred_id, _ = line.split('\t')
            if idx not in predictions:
                predictions[idx] = [pred_id]
            else:
                predictions[idx].append(pred_id)
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate file with predictions for DR evaluation.')
    parser.add_argument('--original_dev', required=True, default='', help='Original dev file with claims.')
    parser.add_argument('--predictions', required=True, default='', help='Anserini predictions file.')
    parser.add_argument('--output', required=True, default='', help='Outpu path.')

    args = parser.parse_args()

    print(f"Reading original dev file from {args.original_dev}...\nReading Anserini predictions from {args.predictions}")
    dev = load_jsonl(args.original_dev)
    predicted = load_anserini_predictions(args.predictions)

    print(f"Saving predictions in {args.output}")
    with open(args.output, 'w') as fw:
        for claim in dev:
            claim['predicted_pages'] = predicted[str(claim['id'])]
            fw.write(json.dumps(claim, ensure_ascii=False) + '\n')
    print(f'Finished')