from __future__ import annotations

from pathlib import Path
from typing import List, Iterator

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Config

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train_tokenizer(
    texts: List[str] | Iterator[str],
    config: Config,
    save_path: Path | None = None,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=config.tokenizer_vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"[BOS]:0 $A:0 [EOS]:0",
        pair=f"[BOS]:0 $A:0 [EOS]:0 [BOS]:1 $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", bos_token_id),
            ("[EOS]", eos_token_id),
        ],
    )

    tokenizer.enable_padding(pad_id=PAD_ID, pad_token="[PAD]")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path / "tokenizer.json"))

    return tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))


def encode(tokenizer: Tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text).ids


def decode(tokenizer: Tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids)