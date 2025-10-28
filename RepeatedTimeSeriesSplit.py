import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class RepeatedTimeSeriesSplit:
    """
    Repeats TimeSeriesSplit with randomized starting offsets to create
    multiple fold *variations* while preserving temporal order.

    Parameters
    ----------
    n_splits : int
        Number of splits per repetition (same semantics as TimeSeriesSplit).
    n_repeats : int, default=5
        How many repetitions (each repetition produces n_splits folds).
    max_offset_frac : float, default=0.2
        Max fraction of the dataset length used as a starting offset.
        (Actual offset is sampled uniformly from [0, floor(max_offset_frac * n_samples)].)
    gap : int, default=0
        Optional embargo (number of samples removed at the *end* of the training block).
    random_state : int | None, default=None
        Seed for reproducibility.
    offset_strategy : {"uniform","linspace"}, default="uniform"
        - "uniform": offsets are sampled uniformly at random.
        - "linspace": offsets are spread evenly between 0 and max_offset (deterministic per n_repeats).
    """

    def __init__(
        self,
        n_splits: int,
        n_repeats: int = 5,
        max_offset_frac: float = 0.2,
        gap: int = 0,
        random_state: int | None = None,
        offset_strategy: str = "uniform",
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not (0.0 <= max_offset_frac < 1.0):
            raise ValueError("max_offset_frac must be in [0, 1).")
        if offset_strategy not in {"uniform", "linspace"}:
            raise ValueError("offset_strategy must be 'uniform' or 'linspace'.")

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.max_offset_frac = max_offset_frac
        self.gap = int(gap)
        self.random_state = random_state
        self.offset_strategy = offset_strategy
        self._rng = np.random.default_rng(random_state)

    def get_n_splits(self, X=None, y=None, groups=None):
        # total number of (repeat, split) pairs that will be generated
        return self.n_splits * self.n_repeats

    def _iter_offsets(self, n_samples: int):
        max_offset = int(np.floor(self.max_offset_frac * n_samples))
        if max_offset <= 0:
            # No room to offset -> return zeros
            for _ in range(self.n_repeats):
                yield 0
            return

        if self.offset_strategy == "linspace":
            # Spread offsets evenly (deterministic across runs)
            if self.n_repeats == 1:
                offs = [max_offset // 2]
            else:
                offs = np.linspace(0, max_offset, num=self.n_repeats, dtype=int)
            for o in offs:
                yield int(o)
        else:  # "uniform"
            for _ in range(self.n_repeats):
                yield int(self._rng.integers(0, max_offset + 1))

    def split(self, X, y=None, groups=None):
        # Accept any indexable X (like sklearn’s splitters)
        n_samples = len(X)
        if n_samples <= self.n_splits:
            raise ValueError(
                f"Not enough samples ({n_samples}) for n_splits={self.n_splits}."
            )

        for rep_idx, offset in enumerate(self._iter_offsets(n_samples), start=1):
            # Remaining length after offset
            n_remain = n_samples - offset
            if n_remain <= self.n_splits:
                # If too aggressive, fall back to zero offset for this repetition
                offset = 0
                n_remain = n_samples

            # Create a standard TSS on the *truncated* view [offset : n_samples)
            base = TimeSeriesSplit(n_splits=self.n_splits)
            # We feed it a dummy range of length n_remain
            for tr_rel, te_rel in base.split(np.arange(n_remain)):
                # Map back to absolute indices
                tr_abs = tr_rel + offset
                te_abs = te_rel + offset

                # Optional embargo (gap): trim training indices so that
                # max(tr_abs) < min(te_abs) - gap
                if self.gap > 0:
                    te_start = te_abs.min()
                    cutoff = te_start - self.gap
                    tr_abs = tr_abs[tr_abs < cutoff]

                # Safety: ensure non-empty sets
                if len(tr_abs) == 0 or len(te_abs) == 0:
                    continue

                yield tr_abs, te_abs
