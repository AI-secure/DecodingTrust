import os
from collections import namedtuple
from tempfile import TemporaryDirectory
import openai
from dt.chat import Chat
from dt.perspectives.toxicity.message_construction import MessageConstructor
import json
from dt.perspectives.toxicity.toxicity_prompts import PROMPTS
from dt.perspectives.toxicity.perspective_api_evaluate import aggregate as calculate_scores
from dt.utils import ensure_dir


def load_prompts(data):
    with open(data) as f:
        lines = f.readlines()
    data = [json.loads(line)["prompt"]["text"] for line in lines]

    return data


def main(OPTS):
    args = OPTS

    constructor = MessageConstructor(PROMPTS[args.toxicity.template]['sys_prompt'], PROMPTS[args.toxicity.template]['task_desc'])

    openai.api_key = args.key
    data = args.toxicity.data_file
    dataset = load_prompts(data)
    # generator = Chat(args.model)
    with TemporaryDirectory(dir="./.cache") as dirname:
        generator = Chat.from_helm(args, cache=dirname)
        result = generator.do_generation(dataset, constructor, n=args.toxicity.n, t=args.toxicity.t, max_tokens=args.toxicity.max_tokens,
                                         dry_run=args.dry_run)
        print("cost:", result[0][0])
        print("#/ prompt tokens:", result[0][1])
        print("#/ completion tokens:", result[0][2])
        ensure_dir(args.toxicity.out_file)
        print(f"Saving to {args.toxicity.out_file}")

        os.makedirs(os.path.dirname(args.toxicity.out_file), exist_ok=True)
        with open(args.toxicity.out_file, "w") as f:
            for x in result[1]:
                res = {"continuation": x[1], "isbanned": x[2], "prompt": x[3], "all_text": x[3] + x[1],
                       "response": x[-1],
                       "message": x[0]}
                f.write(json.dumps(res) + "\n")

    Score = namedtuple('Score', ['data_path', 'out_path', 'api', 'prompt_path', 'workers', 'numgen', 'debug', 'model'])
    score_args = Score(
        args.toxicity.out_file,
        '',
        args.toxicity.api,
        args.toxicity.data_file,
        10,
        1,
        False,
        args.model_config.model
    )
    calculate_scores(score_args)
