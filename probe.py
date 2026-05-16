"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lhs_factor = 0.04087
        clf_in_dim = 44
        self.lin0 = nn.Linear(896, clf_in_dim)
        self.lin1 = nn.Linear(10, clf_in_dim)
        self.lin4 = nn.Linear(24, clf_in_dim)
        self.sp_lhs_coef = nn.Linear(2, 1)
        self.sp_lhs_coef.load_state_dict(OrderedDict([('weight', torch.tensor([[1., 0.]])), ('bias', torch.tensor([-2.]))]))
        self.sp_ent_coef = nn.Parameter(torch.tensor([[0.0]]))
        self.sp_chk_coef = nn.Linear(2, 1)
        self.sp_chk_coef.load_state_dict(OrderedDict([('weight', torch.tensor([[1., 0.]])), ('bias', torch.tensor([-2.]))]))
        self.clf = nn.Linear(clf_in_dim + 2, 2)

    def forward(self, x):
        lhs, ent, sp, chk = x[:,:896], x[:,896:905], x[:,905:907], x[:,907:]
        lhs_x = self.lin0(lhs) * F.sigmoid(self.sp_lhs_coef(sp)) * self.lhs_factor
        ent_x = self.lin1(torch.cat((ent, sp[:,1:]), dim=1)) * (1.0 - sp[:,:1]) * (1.0 - F.sigmoid(self.sp_ent_coef) * sp[:,1:])
        chk_x = self.lin4(chk) * F.sigmoid(self.sp_chk_coef(sp)) 
        a = F.elu(lhs_x + ent_x + chk_x)
        h = torch.cat((a, sp), dim=1)
        logits = self.clf(h)
        return logits


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(self) -> None:
        super().__init__()
        self._net: Net | None = None  # built lazily in fit()
        self._scaler = StandardScaler()
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        self._net = Net()

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,2)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            2-D tensor of raw (pre-softmax) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X_scalable = np.concatenate((X[:,:905], X[:,907:]), axis=1)
        X_scaled = self._scaler.fit_transform(X_scalable)
        X_scaled = np.concatenate((X_scaled[:,:905], X[:,905:907], X_scaled[:,905:]), axis=1)

        self._build_network(X_scaled.shape[1])

        X_t = torch.from_numpy(X_scaled).float()
        y_t = torch.from_numpy(y.astype(int))
        criterion = nn.CrossEntropyLoss()

        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the training loop below.
        # ------------------------------------------------------------------
        optimizer = torch.optim.Adam(self.parameters(), lr=1.55556e-4)

        self.train()
        for _ in range(235):
            optimizer.zero_grad()
            logits = self(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()
        # ------------------------------------------------------------------

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scalable = np.concatenate((X[:,:905], X[:,907:]), axis=1)
        X_scaled = self._scaler.fit_transform(X_scalable)
        X_scaled = np.concatenate((X_scaled[:,:905], X[:,905:907], X_scaled[:,905:]), axis=1)
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.softmax(logits, dim=1).numpy()
        return prob_pos