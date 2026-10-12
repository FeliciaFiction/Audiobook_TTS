"""How much of Kokoro's warm forward pass is the PL-BERT (attention)?

Measures, on a realistic max-size chunk (510 phoneme tokens):

1. full warm forward (median of N, inference_mode)
2. forward with model.bert stubbed to return a cached output tensor
   (everything else identical, so the delta = exact in-context BERT cost)

Also reports the ALBERT's attention implementation as chosen by
transformers, and attention's theoretical FLOP share.

    python bench_bert_share.py [n_tokens]
"""

import sys
import time

import torch

from kokoro_synth import KokoroSynthesizer


def med(fn, reps=9):
    torch.xpu.synchronize()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 510

    synth = KokoroSynthesizer(device="xpu", lang_code="b")
    synth.fast_stft = True
    synth.load_pipeline()
    model = synth.pipeline.model
    print("bert attn implementation:",
          getattr(model.bert.config, "_attn_implementation", "?"))

    torch.manual_seed(0)
    vocab_ids = list(model.vocab.values())
    body = [vocab_ids[i % len(vocab_ids)] for i in range(n_tokens - 2)]
    input_ids = torch.LongTensor([[0, *body, 0]]).to(model.device)
    ref_s = torch.randn((1, 256), device=model.device)

    with torch.inference_mode():
        for _ in range(3):  # warm the shape
            audio, _ = model.forward_with_tokens(input_ids, ref_s, 0.85)
        torch.xpu.synchronize()

        t_full = med(lambda: model.forward_with_tokens(input_ids, ref_s, 0.85))
        secs = audio.shape[-1] / 24000
        print(f"\n{n_tokens} tokens -> {secs:.1f}s audio; "
              f"full warm forward: {t_full*1000:.1f} ms "
              f"({secs/t_full:.0f}x realtime warm)")

        # Capture BERT output and stub it out.
        input_lengths = torch.full((1, 510), 510, device=model.device,
                                   dtype=torch.long)
        text_mask = torch.zeros((1, 510), dtype=torch.bool,
                                device=model.device)
        with torch.no_grad():
            cached = model.bert(input_ids,
                                attention_mask=(~text_mask).int())
        cached = cached.clone()
        real_bert = model.bert

        class BertStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.device = real_bert.device

            def forward(self, ids, attention_mask=None):
                return cached

        model.bert = BertStub()
        t_sans = med(lambda: model.forward_with_tokens(input_ids, ref_s, 0.85))
        model.bert = real_bert

        delta = (t_full - t_sans) * 1000
        print(f"forward with bert stubbed: {t_sans*1000:.1f} ms")
        print(f"BERT cost in-context: {delta:.1f} ms "
              f"({delta/(t_full*1000)*100:.1f}% of forward)")

        # Attention FLOP share within BERT (12 layers, d=768, ffn=2048)
        S = n_tokens
        attn_flops = 12 * 2 * 2 * S * S * 768
        mlp_flops = 12 * 2 * 2 * S * 768 * 2048
        print(f"attention FLOP share inside BERT: "
              f"{attn_flops/(attn_flops+mlp_flops)*100:.1f}% "
              f"(S={S}, so even a free attention kernel saves ~"
              f"{attn_flops/(attn_flops+mlp_flops)*100:.0f}% of BERT)")


if __name__ == "__main__":
    main()
