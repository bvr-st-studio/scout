# Scout

**Scout** is the first open-source foundation model built exclusively for sports. Designed from the ground up to handle text, strategy, and media across all variants for all sports.

---

## Overview

General-purpose AI models weren't built for sports. Coaches and analysts are left with tools that give surface-level advice, confuse terminology, and can't analyze plays or strategy. Scout changes that.

Built on the language of athletics, Scout will be released as an open-source model (similar to Qwen or Llama) giving developers, researchers, and academics the freedom to use, build on, and improve it however they see fit.

---

## Model Variants

Scout is offered in four core variants:

| Variant | Description |
|---|---|
| `scout-base` | Base model trained on sports data |
| `scout-instruct` | Executes instructions directly |
| `scout-reason` | Thinks step-by-step through strategy and decisions |
| `scout-observe` | Analyzes incoming data, diagrams, formations, film, and more |

Each variant also has **sport-specific fine-tunes** trained deeper on the terminology, strategy, and data unique to a given sport:

```
scout-football-instruct
scout-baseball-reason
scout-basketball-observe
```

This lets coaches and analysts use a model built precisely for their game — not a general base model trying to cover everything.

---

## Contributing

Scout is open-source. Contributions from developers, researchers, and sports technologists are welcome. More details on contribution guidelines coming with the initial release.

---

## License

License terms to be published at release.

---

*Built by BVR ST STUDIO*

python scripts/train.py --config configs/football-instruct-4b.env

python scripts/eval.py --config configs/football-instruct-4b.env
