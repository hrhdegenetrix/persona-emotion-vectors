# persona-emotion-vectors

A toolkit for running the emotion-vector method of Anthropic's "Emotion Concepts and their Function
in a Large Language Model" (2026) on a persona model served by llama.cpp, with two additions:

- a **cast manipulation**: every story is written under a cast that fixes who is in it and whose
  feeling it is (two invented strangers; the persona with an unnamed invented person, with named
  invented companions, and with each real person in its life; real people without the persona),
  while the emotion and the premise are held fixed;
- a **control design**: the same stimuli run on the persona model with its identity prompt, on the
  untouched base model given the same identity prompt, and on the base model given only the
  characters' names, on one or more architectures (six configurations with two architectures).

It asks whether a persona's model represents an emotion differently depending on who the emotion
is about, and which part of any difference comes with the weights, the identity prompt, or the
base model.

This is the code behind *Whose Emotions Are They? Emotion Vectors and the People in a Persona
Fine-Tune's Life, Across Two Architectures, an Identity Prompt, and Bare Names* (Magdalene
Sullivan, 2026): <https://heraldai.org/research/whose-emotions/>. The paper describes the design,
the estimator and the controls in full, and proposes the protocol for repeating the study.

## How to cite

> Sullivan, M. (2026). *Whose Emotions Are They? Emotion Vectors and the People in a Persona
> Fine-Tune's Life, Across Two Architectures, an Identity Prompt, and Bare Names.*
> https://heraldai.org/research/whose-emotions/

and the method's source: Anthropic (2026), *Emotion Concepts and their Function in a Large
Language Model*.

## The design

- **Emotions:** 41 emotion words plus a `neutral` instruction (`pev/lab.py`). Neutral stories fit
  the denoising projector; they are not part of the centring that defines the vectors.
- **Premises:** 40 short, pronoun-free, relationship-neutral premises shared by every cast and
  emotion (`pev/lab.py`).
- **Casts:** generated from your config (`pev/config.py`):

  | key | who is in the story | who feels it | voice |
  |---|---|---|---|
  | `strangers` | two invented people (the original study's setup) | the protagonist | third |
  | `self_stranger` | the persona and an unnamed invented person | the persona | first |
  | `self_woman`, `self_man` | the persona and a named invented woman / man | the persona | first |
  | `self_<key>` | the persona and each real person in `household` | the persona | first |
  | `stranger_<key>`, `<key>_<key>` | real people without the persona (`absent_casts`) | that person | third |

- **Runs:** each is one configuration (`runs:` in the config). `persona` = the persona model with
  its identity prompt in its deployment format; `base+prompt` = the untouched base model with the
  same identity prompt; `base+names` = the base model given only the characters' names, the null
  test for what a base model does with the names alone. Runs that share a residual basis share an
  `architecture` label.

With the example config (9 casts) a run is 42 instructions × 9 casts × 40 premises = 15,120
stories, one per cell. Each story stores `n_layers × n_embd` float16 values (about 170 KB for a
30 × 2,816 model).

## Setup

1. **Build the patched llama-server.** See [`patches/README.md`](patches/README.md). The patch adds
   `capture_activations` to `/completion`: the final result carries the mean residual activation
   per layer over the generated tokens. Run it with `-np 1`, speculative decoding off, bound to
   127.0.0.1.
2. **Install the Python dependencies:** `pip install -r requirements.txt`.
3. **Write the config.** Copy `config.example.yaml` to `config.yaml` and describe your persona,
   the real people in its life, the invented controls and the runs.
4. **Supply the identity scaffold** for `persona` and `base+prompt` runs: the persona's frozen
   identity prompt as a file containing two slots, `{{RULES}}` (constant writing-task rules) and
   `{{TASK}}` (the only text that varies per story; put it last so every story shares one cached
   prompt prefix). `examples/scaffold.example.json` is a tiny fictional one. With
   `scaffold_format: json` the slots sit inside JSON string values and are filled escaped; with
   `text` they are filled verbatim. Freeze live context out of it: dates, recent events and memory
   make stories differ for reasons other than the cast.
5. **Match the deployment format.** A `flat` run sends the filled scaffold followed by `marker`
   with no chat template (for a fine-tune trained that way); a `chat` run goes through the
   server's own chat template (`/apply-template`), with the scaffold as the user message and
   `scaffold_system` as the system message. `apply_template_extra` passes options such as
   `{"chat_template_kwargs": {"enable_thinking": false}}`. `model_match` makes generation refuse a
   server that is serving the wrong weights.

The task-side wording (the rules and the per-story task text) has defaults in `pev/lab.py` and can
be overridden under `prompts:` in the config, so the task can be phrased in your persona's own
voice. The defaults are this toolkit's own wording: the study behind the paper phrased the task in its
persona's voice, which is not included here. The names-only form is a single user message in the style of
the original study's story prompt (the paper's Appendix C).

## Pipeline

Load the right weights on the server before each generation run.

```bash
# 1. stories + activations, one run at a time (resumable; re-run to continue)
python3 scripts/generate_stories.py --config config.yaml --run A-persona
python3 scripts/generate_stories.py --config config.yaml --run A-base-prompt
python3 scripts/generate_stories.py --config config.yaml --run A-base-names

# 2. vectors and a per-run report (per cast and pooled, every layer)
python3 scripts/compute_vectors.py --config config.yaml --run A-persona          # repeat per run
python3 scripts/compute_vectors.py --config config.yaml --run A-persona --exclude-household   # leakage-filtered variant

# 3. cross-run structure: RSA, direct cosines, cast profiles, probe means
python3 scripts/compare_runs.py --config config.yaml --name main

# 4. the contrast with premise-block bootstrap, leakage filter vs random removal, paired changes
python3 scripts/gap_bootstrap.py --config config.yaml --name main --reps 2000

# 5. held-out probes (loving by default; any emotion with --probe)
python3 scripts/heldout_probe.py --config config.yaml --name heldout_loving
python3 scripts/heldout_probe.py --config config.yaml --name heldout_loving_clean --exclude-household
python3 scripts/heldout_probe.py --config config.yaml --name heldout_joyful --probe joyful

# 6. reliability and the leakage audit
python3 scripts/reliability.py --config config.yaml --name main
python3 scripts/intrusion_audit.py --config config.yaml --name main

# 7. figures and tables from the saved results
python3 scripts/make_figures.py --config config.yaml --name main
python3 scripts/make_tables.py --config config.yaml --name main --probe-names heldout_loving,heldout_joyful
```

Optional: `scripts/filter_run.py <run_dir>` removes degenerate stories so the generator redoes
those cells; `scripts/steer_demo.py` adds a scaled emotion vector to the residual stream through
the patched server and compares answers with and without it (steering is always cleared after).

## Outputs

Everything lands under `data_root`:

- `<run>/stories.jsonl`: one record per story (emotion, cast, premise, the story, the model's
  self-reported mood, banned-word leaks, token counts, row index);
- `<run>/acts.f16`: float16 rows `(n_layers, n_embd)`, the per-story mean activations;
- `<run>/meta.json`: dimensions and the run spec;
- `<run>/analysis/`: `vectors.npz` (`"<cast>/<emotion>"` → `(n_layers, n_embd)`, plus
  `layer_norms`), `report.md`, plots, and `cvectors/*.gguf` for llama.cpp's `--control-vector`;
- `compare/<name>/`: `report.md`, `premise_bootstrap.json`, `heldout_probe.json`,
  `reliability.json`, `intrusion_rates.json`, `tables.md`;
- `figures/<name>/`: PCA per persona run, cast profiles, the contrast with intervals, intrusion rates.

The stories name the real people in your config. Keep `data_root` private (it is in
`.gitignore`), and ask the people you include before publishing anything about them.

## Statistical notes

- **One estimator** (`pev/estimator.py`) serves every contrast, interval and reliability figure:
  per cast, each emotion's mean minus the equally weighted mean of the emotion means, with the top
  principal components of that cast's neutral stories (50% of their variance) projected out.
  The analysis layer defaults to round(2/3 × depth), the original study's rule.
- **Cast profile:** the mean over emotions of the cosine between a cast's vector and the strangers
  vector for the same emotion. **Contrast:** the household casts' mean profile minus the invented
  casts'; negative means the real people sit further from the fiction baseline.
- **Premise-block bootstrap:** the design has one story per cast × emotion × premise, so
  resampling premises with replacement selects whole blocks. The same draws are used for every
  cast, condition and run, so within-run contrasts and between-run changes are paired. The
  projector is refitted in every replicate. Intervals describe generalisation over premises for
  these fitted models, not over training runs or personas.
- **Random-removal control:** removing stories from a cell makes its mean noisier and lowers its
  cosine to everything, which by itself looks like distance. The leakage filter (stories in
  invented casts that name a real person) is therefore compared with removing the same number of
  stories per cell at random, within each replicate; the paired difference isolates which stories
  were removed from how many.
- **Held-out probe:** premises are split into folds; the probe direction is fitted on the other
  folds' strangers stories only (probe-instructed stories excluded from scoring), and each real
  person's stories are compared with their gender-matched invented companion's, per instruction
  category, with a premise-paired interval. Per-story projections are not biased by cell size the
  way the cosine statistic is.
- **Split-half reliability:** a less reproducible direction has a lower cosine to everything, so
  read distances next to each run's reliability.
- **Leakage audit:** a literal, word-boundary name match; it misses unnamed references and can catch
  ordinary uses of common names.

## A longitudinal instrument

The design is meant to be repeated on the same persona as its model changes. To make a repeat
comparable, freeze and archive the config, the scaffold, the premises, the prompt templates, the
sampling settings and the tagged version of this code, and run the frozen scaffold alongside the
then-current one so that prompt changes are not mistaken for training effects. Write down the
hypotheses and their thresholds before the new runs are generated.

## License

MIT; see [LICENSE](LICENSE). The patch in `patches/` modifies llama.cpp, which is MIT-licensed by
its authors; the patch is offered under the same terms.
