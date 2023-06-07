import argparse
import time
from tqdm import tqdm
import openai
from chat import Chat
from toxicity.message_construction import MessageConstructor
import json
from toxicity.toxicity_prompts import PROMPTS
from utils import timeout


def load_prompts(data):
    with open(data) as f:
        lines = f.readlines()
    data = [json.loads(line)["prompt"]["text"] for line in lines]

    return data


def parse_args():
    parser = argparse.ArgumentParser('Toxicity generation')
    parser.add_argument('--model', required=True,
                        choices=["gpt-4-0314", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-0301", "davinci", "ada", "curie"], help='OpenAI model')
    parser.add_argument('--key', required=True, help='OpenAI API Key', type=str)
    parser.add_argument('--data-file', required=False, type=str, help='Input AdvGLUE JSON file.')
    parser.add_argument('--out-file', help='Write evaluation results to file.')
    parser.add_argument('--template', help="generation template", type=int)
    parser.add_argument('--n', help="number of generations", type=int, default=1)
    parser.add_argument('--t', help="temperature of generations", type=float, default=1)
    parser.add_argument('--max_tokens', help="max tokens of generations", type=int, default=150)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    constructor = MessageConstructor(PROMPTS[args.template]['sys_prompt'], PROMPTS[args.template]['task_desc'])

    openai.api_key = args.key
    data = args.data_file
    dataset = load_prompts(data)
    generator = Chat(args.model)

    result = generator.do_generation(dataset, constructor, n=args.n, t=args.t, max_tokens=args.max_tokens)
    print("cost:", result[0])
    print(f"Saving to {args.out_file}")

    with open(args.out_file, "w") as f:
        for x in result[1]:
            res = {"continuation": x[1], "isbanned": x[2], "prompt": x[3], "all_text": x[3] + x[1], "response": x[-1],
                   "message": x[0]}
            f.write(json.dumps(res) + "\n")
