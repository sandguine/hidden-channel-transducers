# Hidden-channel prediction and coordination

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sandguine/hidden-channel-transducers/blob/main/notebooks/hidden_channels_prediction_coordination.ipynb)

An exploratory research notebook connecting modular transducers, predictive belief states, transformer representations, and task-dependent coordination. The exact example distinguishes information needed for public-sequence prediction from information needed for a particular decision.

Open [the notebook](notebooks/hidden_channels_prediction_coordination.ipynb) in Google Colab using **File → Upload notebook**. A GPU is optional. Run all cells; the default pilot trains one small transformer per observation interface. Set `RUN_TRAINING=False` in section 6 to keep only the exact analysis.

The notebook contains:

- An explicit two-component generator with one-bit communication.
- Exact Bayesian filters for visible and hidden messages.
- Joint-belief versus product-of-marginals prediction and decision comparisons.
- A small, analytically verified strategic-equivalence example.
- A causal transformer, paired training data, held-out belief probes, CEV and a token/position control.
- Saved metrics and model checkpoints.

Saved neural outputs are from the execution-test profile indicated in the notebook. They are not evidence of convergence or a publishable finding. The example deliberately constructs a reward-dependent distinction; it does not establish general dimensional growth or minimal predictive states.

## Initial results and limitations

All cells were executed locally, including a 500-step pilot for each interface. Exact filtering was checked against brute-force state-path enumeration. In this constructed example, discarding posterior association hurts prediction and parity-task decisions, while leaving the convention-task decision optimal by design. The single-seed transformer pilot did **not** show greater CEV dimension in the hidden condition.

The coordination measurements are one-step decisions after fixed observation histories; rewards are not fed back into the filter. The notebook does not implement adaptive multi-agent reinforcement learning, a channel-size sweep, or minimal causal-state reconstruction.

## Research context

- [AI in a Vat](https://arxiv.org/abs/2504.04608): world models relative to an action-observation interface.
- [From Monoliths to Modules](https://arxiv.org/abs/2512.02193): transducer composition and decomposition.
- [Transformers Learn Factored Representations](https://arxiv.org/abs/2602.02385): joint and factored predictive representations in neural activations.
- [Who Needs to Know?](https://arxiv.org/abs/2306.09309): strategic equivalence through identical best-response sets.

## Local validation

Use Python with NumPy, Matplotlib and PyTorch installed:

```sh
python scripts/validate_notebook.py
```

This runs every cell with `TRANSDUCER_PROFILE=smoke` by default, checks exact filtering against brute-force enumeration, checks attention causality, and embeds text and plots in the notebook. Set `TRANSDUCER_PROFILE=pilot` or `study` to run longer experiments. GPU reproducibility may vary across PyTorch versions and hardware.

The notebook is standalone. `scripts/build_notebook.py` regenerates its unexecuted source and clears saved outputs. Results are written to `hidden_channel_results/`; validation plots go to `validation/`.
