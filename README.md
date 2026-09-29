# MP1 code — installation and usage

Read [the project guide](https://chatgpt.com/guide/GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules. The package has only these two documents.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU.

After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Final submitted model** — width 256, depth 4, four attention heads, context 256 and SwiGLU hidden size 341:

```bash
python train.py --implementation student --config configs/width256.json --device cpu --threads 4 --seed 17 --steps 4800 --eval-every 300 --run-dir runs/width256_qknorm_hidden512_4800
```

The final checkpoint was selected using the validation split and then frozen before final test evaluation.

Final test evaluation:

```bash
python evaluate.py --checkpoint runs/width256_qknorm_hidden512_4800/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

The final submitted model has **2,633,384 parameters**. Its final validation BPB is **1.6077307845** and its complete-test BPB is **1.6304622955**.

The final test was run with FP32 on CPU using four threads. The measured scoring time was **17.50 seconds**. Peak RAM was measured separately at **1.514 GiB**.

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB.

Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files                               | Use                                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------------------- |
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons.                                        |
| `student.py`, `train.py`            | Submitted model factory and training recipe.                                        |
| `configs/width256.json`             | Final submitted model configuration.                                                |
| `common.py`, `evaluate.py`          | Fixed data checks, windows and scorer; keep unchanged.                              |
| `data/`                             | Supplied splits, tokenizer and dataset hashes; keep unchanged.                      |
| `tests/test_contract.py`            | Checks your model's causality, normalization, independence and gradients.           |
| `RUN_LOG_TEMPLATE.csv`              | Optional experiment-log template.                                                   |
| `PACKAGE_MANIFEST.json`             | Release hashes; paths are relative to the package root containing code/ and guide/. |

The final checkpoint bundle contains the frozen predictor checkpoint and the associated evaluation/metrics files.

- `build_model(config)` returns a PyTorch model with `context=256`.

- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.

- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.

- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.

- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split      | Scored targets | UTF-8 bytes |
| ---------- | -------------- | ----------- |
| Validation | 376,599        | 1,148,007   |
| Test       | 428,405        | 1,292,013   |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

For the final frozen predictor:

| Resource         | Measured      | Limit                                  |
| ---------------- | ------------- | -------------------------------------- |
| CPU scoring time | **17.50 s**   | ≤ 29.60 s (5× baseline)                |
| Peak RAM         | **1.514 GiB** | ≤ 4 GiB                                |
| Checkpoint size  | **10.07 MiB** | ≤ 64 MiB uncompressed inference assets |

The final test score was obtained with FP32 CPU evaluation under protocol `7506-mp1-wt2-v2`.

## 5. Prepare your submission and reproduce a peer

The [guide](https://chatgpt.com/guide/GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references**

- **Reproduction instructions**

- **The final model implementation and required configuration**

- **The matching complete checkpoint bundle**

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link. The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.

## 7. AI assistance disclosure

AI assistance was used during development for debugging, code review, explanation of machine-learning concepts, experiment planning, and preparation of documentation. The submitted implementation, training configuration, experimental decisions and final results were reviewed and verified by the author. No external pretrained weights, external datasets, cached test answers or other prohibited evaluation information were used.
