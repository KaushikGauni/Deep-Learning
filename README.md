# Deep Learning — Assignments

Course assignments for M.Tech Deep Learning. Each notebook is self-contained: it loads a dataset,
builds a model, trains it, and reports accuracy / BLEU plus a confusion matrix or loss curve.

Assignment 1 is **pure NumPy** — forward pass, backprop and optimizers all written by hand, no
framework. Assignment 2 moves to **PyTorch** and runs locally. Assignment 3 mixes
**TensorFlow/Keras** (image captioning) and **PyTorch** (BERT, translation), and is written to run
on **Google Colab**.

---

## Assignment 1 — Neural networks from scratch in NumPy

No autograd, no `nn.Module`. Every gradient is derived and coded by hand. All three tasks train a
fully-connected network on tabular CSV data and compare training variants side by side.

| Notebook | What it does |
|---|---|
| [Assignment 1 - Task 1.ipynb](<Assignment 1 - Task 1.ipynb>) | MLFFNN + backprop from scratch, comparing **5 optimizers** |
| [Assignment 1 - Task 2.ipynb](<Assignment 1 - Task 2.ipynb>) | Same network, comparing **batch vs layer normalization** |
| [Assignment 1 - Task 3.ipynb](<Assignment 1 - Task 3.ipynb>) | Deep FNN: **supervised vs autoencoder pretraining** |

### Task 1 — Backprop and optimizer comparison
A 3-layer MLFFNN (`input → 20 → 10 → num_classes`) written entirely in NumPy: `tanh` and its
derivative, numerically-stable `softmax`, and cross-entropy all hand-coded, with the backward pass
derived layer by layer (`dz3 = ŷ - y`, then `da = W·dz`, `dz = da ⊙ (1 - a²)` back through both
hidden layers). Features are standardised with `StandardScaler`. Training is **stochastic** —
one sample at a time, `np.outer` for weight gradients.

Five optimizers are implemented from their update rules and compared at `lr=0.01`:

| Optimizer | Update rule |
|---|---|
| **SGD** | `W ← W - lr·g` |
| **Momentum** | `v ← αv - lr·g`, `W ← W + v` (α = 0.9) |
| **AdaGrad** | `cache ← cache + g²`, `W ← W - lr·g/(√cache + ε)` |
| **RMSProp** | `cache ← βcache + (1-β)g²`, same step (β = 0.9) |
| **Adam** | momentum + RMSProp with bias correction (β₁ = 0.9, β₂ = 0.999) |

Training stops when the epoch-to-epoch change in average error drops below `1e-4`, capped at 200
epochs. Each optimizer gets side-by-side train/test confusion matrices, and a final chart overlays
all five error curves plus the epochs each needed to converge.

### Task 2 — Batch vs layer normalization
Same `input → 20 → 10 → classes` architecture, but **vectorised**: mini-batches of 64, matrix
forward/backward passes, one-hot targets, and Adam only. The point of the task is the `normalize()`
function, which takes the pre-activation `Z` and standardises it either **across features within a
sample** (layer norm, `axis=1`) or **across the batch for each feature** (batch norm, `axis=0`) —
the difference is documented with a worked example in the notebook.

Three runs — no normalization, layer norm, batch norm — all start from the *same* seeded initial
weights so the comparison is fair. Each reports a train/test confusion matrix (built with a
hand-written `confusion_matrix`), and the notebook finishes with all three loss curves on one axis
plus a table of epochs-to-convergence.

### Task 3 — Supervised vs unsupervised (autoencoder) pretraining
The deepest network in the assignment: `PCA-dim → 256 → 128 → 64 → classes`, sigmoid hidden units,
softmax output, He initialisation, Adam inlined into the training loop, 50 epochs of mini-batch
training. Inputs are standardised and then reduced with **PCA retaining 95% variance**.

Two initialisation strategies are compared:
1. **Supervised** — random (He) init, trained directly on labels.
2. **Unsupervised pretraining** — three autoencoders trained greedily layer by layer
   (`input→256→input`, then `256→128→256`, then `128→64→128`), each reconstructing its own input
   with hand-derived gradients. The learned encoder weights then initialise the classifier's first
   three layers before supervised fine-tuning.

Results are shown as a 2×2 grid of confusion matrices (supervised train/test, pretrained train/test).

---

## Assignment 2 — CNNs and RNNs from scratch

| Notebook | What it does |
|---|---|
| [Assignment 2 - Task 1(a).ipynb](<Assignment 2 - Task 1(a).ipynb>) | Image classification via **VGG16 transfer learning (feature extraction)** |
| [Assignment 2 - Task 1(b).ipynb](<Assignment 2 - Task 1(b).ipynb>) | Image classification via **GoogLeNet transfer learning (feature extraction)** |
| [Assignment 2 - Task 2.ipynb](<Assignment 2 - Task 2.ipynb>) | Image classification with a **CNN trained from scratch** |
| [Assignment 2 - Task 3.ipynb](<Assignment 2 - Task 3.ipynb>) | **Sentiment analysis** on tweets with a vanilla RNN + GloVe |
| [Assignment 2 - Task 4.ipynb](<Assignment 2 - Task 4.ipynb>) | **POS tagging** with a sequence-labelling RNN + GloVe |

### Task 1(a) — VGG16 feature extraction + MLFFNN
Images are resized to 224×224 and ImageNet-normalised. A pretrained `vgg16` is truncated to
`features + avgpool` and used as a **frozen** feature extractor (25088-d vectors). Those features
train a 3-layer MLFFNN (`input → 1024 → 256 → num_classes`, ReLU), Adam @ 5e-4, 20 epochs,
CrossEntropyLoss. Outputs per-epoch loss/accuracy, a confusion matrix, and a training-loss curve.

### Task 1(b) — GoogLeNet feature extraction + MLFFNN
Same pipeline with a different backbone: `googlenet` (ImageNet weights) with `fc` replaced by
`nn.Identity()` and `transform_input = False`, giving 1024-d features. Classifier is
`1024 → 512 → 128 → num_classes` with ReLU + **Dropout(0.5)**, Adam @ 1e-4, 10 epochs. Reports
accuracy per epoch, final accuracy, confusion matrix and loss curve. Together with 1(a) this
compares two pretrained backbones on the same dataset.

### Task 2 — CNN built from scratch
No pretrained weights. Images resized to 64×64 and normalised to [-1, 1]. Architecture follows the
task spec: **CL1** (3→4 feature maps, 3×3) → ReLU → **PL1** (2×2 *average* pooling) → **CL2**
(4→8 maps, 3×3) → ReLU → **PL2** (2×2 average pooling) → flatten → FC to `num_classes`.
Adam @ 1e-3, 10 epochs. Reports train loss/accuracy per epoch, test accuracy, and a seaborn
confusion-matrix heatmap.

### Task 3 — Sentiment analysis (RNN + GloVe)
Binary tweet sentiment (`Positive` / `Negative`) from `train.csv` / `test.csv`. Text is lowercased,
labels mapped to 1/0 and NaN rows dropped. A word-level vocabulary is built from the training set
(`<PAD>`, `<UNK>`), and an embedding matrix is initialised from **GloVe 6B 200d** (words missing
from GloVe get random normal vectors). Sequences are padded/truncated to 100 tokens.

Model: frozen embedding → `nn.RNN(200, 25)` over `pack_padded_sequence` → Dropout(0.2) → linear to
2 classes. Class imbalance is handled with a **weighted CrossEntropyLoss** (weights `[1.0, 2.5]`).
Adam @ 5e-4, 30 epochs. Final report breaks out correct/wrong counts and accuracy **per sentiment
class** plus overall accuracy, with a confusion-matrix heatmap.

### Task 4 — POS tagging (RNN + GloVe)
Sequence labelling: each token in a sentence gets a POS tag. Separate `word2idx` and `tag2idx`
vocabularies are built from the training split, both padded with index 0; embeddings again come
from **GloVe 6B 200d** and are frozen. Sentences are padded/truncated to 50 tokens.

Model: embedding → `nn.RNN(200, 25)` emitting a hidden state **per timestep** → linear to
`tag_size`. Loss is CrossEntropyLoss with `ignore_index=0` so padding contributes nothing, and
accuracy is computed under the same mask. Adam @ 1e-3, 10 epochs. Reports per-epoch loss,
accuracy and error, then test accuracy/error plus a confusion matrix over all tags.

---

## Assignment 3 — Captioning, Transformers, and Machine Translation

| Notebook | What it does |
|---|---|
| [Assignment 3 - Task 1.ipynb](<Assignment 3 - Task 1.ipynb>) | **Image captioning** — VGG16 encoder + **SimpleRNN** decoder |
| [Assignment 3 - Task 2.ipynb](<Assignment 3 - Task 2.ipynb>) | **Image captioning** — VGG16 encoder + **LSTM** decoder |
| [Assignment 3 - Task 3.ipynb](<Assignment 3 - Task 3.ipynb>) | **Sentiment analysis with BERT** fine-tuning |
| [Assignment 3 - Task 4.ipynb](<Assignment 3 - Task 4.ipynb>) | **English → Kannada translation** with an LSTM encoder-decoder |
| [Assignment 3 - Task 5.ipynb](<Assignment 3 - Task 5.ipynb>) | **English → Kannada translation** with a Transformer built from scratch |

### Tasks 1 & 2 — Image captioning (the RNN vs LSTM comparison)
These two notebooks are **deliberately identical except for the decoder recurrent layer** — Task 1
uses `SimpleRNN(50)`, Task 2 uses `LSTM(50)` — so the BLEU scores isolate the effect of gating.

Shared pipeline (Keras, on Colab with Google Drive mounted):
- Captions are cleaned (lowercased, punctuation stripped) and wrapped in `startseq` / `endseq`.
- **VGG16** with the final classification layer removed produces a 4096-d vector per image; features
  are cached to `features.pkl` so extraction runs only once.
- Caption words are embedded with **GloVe 6B 200d** (frozen).
- Training pairs are generated by unrolling each caption: given the image feature + first *i* words,
  predict word *i+1*.
- 90/10 train/test split. Decoder merges image and text branches with `add`, then
  `Dense(50) → Dense(vocab_size, softmax)`. `sparse_categorical_crossentropy`, Adam, 20 epochs.
- Evaluation: greedy caption generation, then **corpus BLEU-1..4** on both train (200 images) and
  test sets, plus 5 sample images per split with ground-truth vs predicted captions.

### Task 3 — BERT fine-tuning for sentiment analysis
Revisits Assignment 2 Task 3's dataset with a pretrained transformer instead of an RNN.
`bert-base-uncased` tokenises tweets to `MAX_LEN=128` with attention masks. The classifier head is
`pooler_output (768) → Linear(100) → ReLU → Dropout(0.3) → Linear(num_classes)`, with **balanced
class weights** (`compute_class_weight`) in the loss. AdamW @ 2e-5, 3 epochs — the whole BERT stack
is fine-tuned, not frozen. Reports accuracy, confusion matrix, `classification_report`, BLEU@1..4
over the label strings, and 5 worked examples from each split.

### Task 4 — English → Kannada NMT (LSTM encoder-decoder)
Parallel corpus `team23_kn_train.csv` / `team23_kn_test.csv`. Vocabularies are built per language
with `SOS/EOS/UNK/PAD` specials and `min_freq=2`; pairs longer than 20 words are filtered out.
English embeddings come from **GloVe 200d**, Kannada from **FastText `wiki.kn.vec`**; both fall back
to random init if the files are missing.

- **Encoder**: single-layer `nn.LSTM` (hidden 100) over packed sequences.
- **Decoder**: single-layer `nn.LSTM` initialised from the encoder's final `(hidden, cell)`, stepped
  one token at a time with **teacher forcing at ratio 0.5**.
- Adam @ 1e-3, 20 epochs, gradient clipping at norm 1.0, `ignore_index=PAD` in the loss.
- BLEU-1..4 is computed with a **hand-rolled clipped n-gram precision** (no NLTK) every 5 epochs;
  the best BLEU-1 checkpoint is saved to `best_model_task4.pt` and reloaded for final evaluation.
- Ends with 5 training and 5 test translations (source / ground truth / prediction) and a saved
  loss-vs-epoch plot (`loss_plot.png`).

### Task 5 — English → Kannada NMT (Transformer from scratch)
Same data, same vocabularies, same embeddings and same BLEU harness as Task 4 — the point is a
like-for-like comparison against the LSTM. The model is a **Transformer implemented by hand**, not
`nn.Transformer`:

- `PositionalEncoding` — sinusoidal, registered as a buffer.
- `MultiHeadAttention` — explicit `W_q/W_k/W_v/W_o`, scaled dot-product, mask-aware.
- `FeedForward`, `EncoderLayer` (self-attn + FFN, post-norm residuals), `DecoderLayer`
  (masked self-attn + cross-attn + FFN).
- 3 encoder layers, 3 decoder layers, 4 heads, `d_model=256`, `d_ff=512`, dropout 0.1.
  200-d pretrained embeddings are projected up to `d_model` and scaled by `sqrt(d_model)`;
  weights use Xavier init.
- Masking: source padding mask, plus a causal (`tril`) mask combined with the target padding mask.
- Adam @ 5e-4 with a **warmup-then-constant LR schedule** (`WARMUP_STEPS=2000`, stepped per batch),
  20 epochs. Best checkpoint saved to `best_model_task5.pt`.
- Same outputs as Task 4: BLEU-1..4 and 5 train + 5 test example translations.

---

## Running the notebooks

**Paths are hard-coded and must be updated before running.** Assignments 1 and 2 point at local
Windows paths (`C:\Users\...\Assignment 1\DataSets\...`); Assignment 3 points at Colab paths
(`/content/...`, `/content/drive/MyDrive/...`).

Datasets and embeddings are **not** in this repo and need to be supplied separately:

- Assignment 1 tabular CSVs: `task1/` and `task2/` (`train_data.csv`, `train_label.csv`,
  `val_*`, `test_*`), `task3/` (`training_data_set_23_labeled_data.csv` and matching
  labels/validation/testing files)
- Image classification dataset in `ImageFolder` layout (`train/<class>/*`, `test/<class>/*`)
- Sentiment CSVs with `tweet` and `sentiment` columns
- POS tagging CSVs with `Sentence` and `Tags` columns (space-separated, aligned)
- Image captioning: `Images/`, `captions.txt`, `image_names.txt`
- Translation: `team23_kn_train.csv`, `team23_kn_test.csv`
- Embeddings: `glove.6B.200d.txt`, `wiki.kn.vec` (FastText Kannada)

### Dependencies
```
numpy  pandas  scikit-learn  matplotlib   # Assignment 1 needs nothing more than this
torch  torchvision  transformers          # Assignment 2, Assignment 3 Tasks 3-5
tensorflow                                # Assignment 3 Tasks 1-2
nltk  seaborn  pillow
```
A GPU is strongly recommended — every notebook selects CUDA when available. Assignment 3 Tasks 1–5
in particular are written for Colab GPU runtimes.
