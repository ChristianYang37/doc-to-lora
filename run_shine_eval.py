import argparse
import logging

from ctx_to_lora.shine_eval import (
    evaluate_qa,
    evaluate_mqa_multiturn,
    evaluate_wikitext_recon_comp,
    load_eval_model,
)


def parse_lengths(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SHINE-style evaluation protocols with doc-to-lora models."
    )
    parser.add_argument(
        "--task",
        choices=["qa", "wikitext_recon_comp", "mqa_multiturn"],
        required=True,
    )
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--gen_lora_scaling", type=float, default=1.0)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Defaults to SHINE's per-task value: qa=128, wikitext/mqa=500.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--source",
        choices=[
            "squad",
            "hotpotqa",
            "musique",
            "2wikimultihopqa",
            "msmarco_v1",
            "msmarco_v2",
        ],
        default="squad",
    )
    parser.add_argument("--context_avg_len", type=int, default=512)
    parser.add_argument("--context_max_length", type=int, default=1300)
    parser.add_argument("--conversation_max_length", type=int, default=128)

    parser.add_argument(
        "--wikitext_data_dir",
        type=str,
        default="data/wikitext/wikitext-2-raw-v1",
    )
    parser.add_argument("--wikitext_split", type=str, default="train")
    parser.add_argument("--wikitext_idx_dict_path", type=str, default=None)
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("1,2,3,4,5,6,7,8,9,10,11"))
    parser.add_argument("--max_samples_per_length", type=int, default=-1)

    parser.add_argument(
        "--mqa_data_path",
        type=str,
        default="data/msmacro-mqa/test.jsonl",
    )
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_conversation_length", type=int, default=3000)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    model, tokenizer = load_eval_model(
        checkpoint_path=args.checkpoint_path,
        model_name_or_path=args.model_name_or_path,
        gen_lora_scaling=args.gen_lora_scaling,
    )

    if args.task == "qa":
        max_new_tokens = 128 if args.max_new_tokens is None else args.max_new_tokens
        evaluate_qa(
            model=model,
            tokenizer=tokenizer,
            source=args.source,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            context_avg_len=args.context_avg_len,
            context_max_length=args.context_max_length,
            conversation_max_length=args.conversation_max_length,
            max_new_tokens=max_new_tokens,
        )
    elif args.task == "wikitext_recon_comp":
        max_new_tokens = 500 if args.max_new_tokens is None else args.max_new_tokens
        evaluate_wikitext_recon_comp(
            model=model,
            tokenizer=tokenizer,
            data_dir=args.wikitext_data_dir,
            output_dir=args.output_dir,
            split=args.wikitext_split,
            lengths=args.lengths,
            idx_dict_path=args.wikitext_idx_dict_path,
            max_samples_per_length=args.max_samples_per_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_new_tokens=max_new_tokens,
        )
    elif args.task == "mqa_multiturn":
        max_new_tokens = 500 if args.max_new_tokens is None else args.max_new_tokens
        evaluate_mqa_multiturn(
            model=model,
            tokenizer=tokenizer,
            data_path=args.mqa_data_path,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=1,
            num_workers=args.num_workers,
            context_max_length=args.context_max_length,
            conversation_max_length=args.conversation_max_length,
            max_new_tokens=max_new_tokens,
            max_conversation_length=args.max_conversation_length,
        )
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
