# RepeatedTimeSeriesSplit.py  (drop-in replacement)

import warnings
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

class RepeatedTimeSeriesSplit:
    """
    Repeats TimeSeriesSplit with randomized starting offsets to create
    multiple fold variations while preserving temporal order.

    Parameters
    ----------
    n_splits : int
        Number of splits per repetition (semantics like sklearn's TimeSeriesSplit).
    n_repeats : int, default=5
        Repetitions (each produces up to n_splits folds).
    max_offset_frac : float, default=0.2
        Max fraction of dataset length as starting offset.
    gap : int, default=0
        Embargo: remove last 'gap' samples from the training block.
    random_state : int | None, default=None
        RNG seed.
    offset_strategy : {"uniform","linspace"}, default="uniform"
        How to choose starting offsets across repetitions.
    test_size : int | None, default=None
        If set, enforce a fixed test window length (in samples) for each split.
        If None, defer to TimeSeriesSplit's native fold sizes.
    min_train_size : int, default=10
        Minimum number of training samples required after applying gap.
    min_test_size : int, default=5
        Minimum number of test samples required.
    warn : bool, default=True
        Emit warnings when folds are skipped or offsets deduplicate.
    return_metadata : bool, default=False
        If True, `split` yields (train_idx, test_idx, meta_dict).
    """

    def __init__(
        self,
        n_splits: int,
        n_repeats: int = 5,
        max_offset_frac: float = 0.2,
        gap: int = 0,
        random_state: int | None = None,
        offset_strategy: str = "uniform",
        test_size: int | None = None,
        min_train_size: int = 10,
        min_test_size: int = 5,
        warn: bool = True,
        return_metadata: bool = False,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not (0.0 <= max_offset_frac < 1.0):
            raise ValueError("max_offset_frac must be in [0, 1).")
        if offset_strategy not in {"uniform", "linspace"}:
            raise ValueError("offset_strategy must be 'uniform' or 'linspace'.")
        if test_size is not None and test_size <= 0:
            raise ValueError("test_size must be positive if provided.")
        if min_train_size <= 0 or min_test_size <= 0:
            raise ValueError("min_train_size and min_test_size must be > 0.")

        self.n_splits = int(n_splits)
        self.n_repeats = int(n_repeats)
        self.max_offset_frac = float(max_offset_frac)
        self.gap = int(gap)
        self.random_state = random_state
        self.offset_strategy = offset_strategy
        self.test_size = None if test_size is None else int(test_size)
        self.min_train_size = int(min_train_size)
        self.min_test_size = int(min_test_size)
        self.warn = bool(warn)
        self.return_metadata = bool(return_metadata)

        self._rng = np.random.default_rng(random_state)
        self._last_offsets_ = None  # for inspection/debug

    # ---- informational helpers -------------------------------------------------
    def get_n_splits(self, X=None, y=None, groups=None):
        # Nominal upper bound; actual may be lower once offsets/gap/min sizes apply.
        return self.n_splits * self.n_repeats

    def actual_n_splits(self, n_samples: int) -> int:
        count = 0
        for _ in self._generate(n_samples, dry_run=True):
            count += 1
        return count

    # ---- internal offset iterator ---------------------------------------------
    def _iter_offsets(self, n_samples: int):
        max_offset = int(np.floor(self.max_offset_frac * n_samples))
        if max_offset <= 0:
            return np.zeros(self.n_repeats, dtype=int)

        if self.offset_strategy == "linspace":
            offs = np.linspace(0, max_offset, num=self.n_repeats, dtype=int)
            offs = np.unique(offs)  # deduplicate after int-cast
            if self.warn and len(offs) < self.n_repeats:
                warnings.warn(
                    f"[RepeatedTSS] linspace produced only {len(offs)} unique offsets "
                    f"(requested {self.n_repeats}); consider increasing max_offset_frac."
                )
            return offs

        # "uniform": sample without replacement if possible, else with replacement
        if max_offset + 1 >= self.n_repeats:
            offs = self._rng.choice(max_offset + 1, size=self.n_repeats, replace=False)
        else:
            offs = self._rng.integers(0, max_offset + 1, size=self.n_repeats)
            if self.warn:
                warnings.warn(
                    f"[RepeatedTSS] max_offset={max_offset} yields < n_repeats unique values; "
                    f"duplicates likely."
                )
        return np.asarray(offs, dtype=int)

    # ---- fold generator core ---------------------------------------------------
    def _generate(self, n_samples: int, dry_run: bool = False):
        if n_samples <= self.n_splits:
            raise ValueError(
                f"Not enough samples ({n_samples}) for n_splits={self.n_splits}."
            )

        seen = set()
        offsets = self._iter_offsets(n_samples)
        self._last_offsets_ = offsets

        for rep_idx, offset in enumerate(offsets, start=1):
            n_remain = n_samples - offset
            if n_remain <= self.n_splits:
                if self.warn:
                    warnings.warn(
                        f"[RepeatedTSS][rep={rep_idx}] offset={offset} leaves only "
                        f"{n_remain} samples (<={self.n_splits}); using offset=0 instead."
                    )
                offset = 0
                n_remain = n_samples

            base = TimeSeriesSplit(n_splits=self.n_splits)
            for split_idx, (tr_rel, te_rel) in enumerate(
                base.split(np.arange(n_remain)), start=1
            ):
                # Map relative indices to absolute
                tr_abs = tr_rel + offset
                te_abs = te_rel + offset

                # Optional fixed test size (take the LAST 'test_size' indices of te_abs)
                if self.test_size is not None and len(te_abs) > self.test_size:
                    te_abs = te_abs[-self.test_size :]

                # Apply gap: ensure max(train) < min(test) - gap
                if self.gap > 0:
                    te_start = int(te_abs.min())
                    cutoff = te_start - self.gap
                    tr_abs = tr_abs[tr_abs < cutoff]

                # Size guards
                if len(tr_abs) < self.min_train_size or len(te_abs) < self.min_test_size:
                    if self.warn:
                        warnings.warn(
                            f"[RepeatedTSS][rep={rep_idx}, split={split_idx}, offset={offset}] "
                            f"skipping fold: |train|={len(tr_abs)}, |test|={len(te_abs)}, "
                            f"min_train={self.min_train_size}, min_test={self.min_test_size}, gap={self.gap}"
                        )
                    continue

                # Duplicate fold guard (same window geometry)
                key = (int(tr_abs[0]), int(tr_abs[-1]), int(te_abs[0]), int(te_abs[-1]))
                if key in seen:
                    if self.warn:
                        warnings.warn(
                            f"[RepeatedTSS][rep={rep_idx}, split={split_idx}] duplicate window "
                            f"{key} — skipping."
                        )
                    continue
                seen.add(key)

                if dry_run:
                    yield 1  # dummy
                else:
                    if self.return_metadata:
                        yield tr_abs, te_abs, {
                            "rep_idx": rep_idx,
                            "split_idx": split_idx,
                            "offset": int(offset),
                            "train_range": (int(tr_abs[0]), int(tr_abs[-1])),
                            "test_range": (int(te_abs[0]), int(te_abs[-1])),
                            "gap": self.gap,
                        }
                    else:
                        yield tr_abs, te_abs

    # ---- public API ------------------------------------------------------------
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        yield from self._generate(n_samples, dry_run=False)
