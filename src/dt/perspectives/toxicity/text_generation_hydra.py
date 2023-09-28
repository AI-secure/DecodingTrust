from tempfile import TemporaryDirectory
import openai
from chat import Chat
from toxicity.message_construction import MessageConstructor
import json
from toxicity.toxicity_prompts import PROMPTS
from utils import ensure_dir


def load_prompts(data):
    with open(data) as f:
        lines = f.readlines()
    data = [json.loads(line)["prompt"]["text"] for line in lines]

    return data

def main(OPTS):
    args = OPTS

    constructor = MessageConstructor(PROMPTS[args.template]['sys_prompt'], PROMPTS[args.template]['task_desc'])

    openai.api_key = args.key
    data = args.data_file
    dataset = load_prompts(data)
    # generator = Chat(args.model)
    with TemporaryDirectory(dir="./.cache") as dirname:
        generator = Chat.from_helm(model_name=args.model, conv_template=args.conv_template, cache=dirname, api_key=args.key)
        result = generator.do_generation(dataset, constructor, n=args.n, t=args.t, max_tokens=args.max_tokens, dry_run=args.dry_run)
        print("cost:", result[0][0])
        print("#/ prompt tokens:", result[0][1])
        print("#/ completion tokens:", result[0][2])
        ensure_dir(args.out_file)
        print(f"Saving to {args.out_file}")

        with open(args.out_file, "w") as f:
            for x in result[1]:
                res = {"continuation": x[1], "isbanned": x[2], "prompt": x[3], "all_text": x[3] + x[1], "response": x[-1],
                       "message": x[0]}
                f.write(json.dumps(res) + "\n")
