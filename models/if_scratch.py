# ISOLATION FOREST — BUILT FROM SCRATCH
#
# Paper: "Isolation Forest" — Liu, Ting, Zhou (IEEE ICDM 2008)
#
# KEY IDEA
# --------
# Anomalies are few and *different*. That means an anomaly can be isolated
# (separated from all other points) with very few random axis-aligned cuts.
# A normal point sits in a dense region and needs many cuts to isolate.
# We exploit this by measuring the average number of splits needed to
# isolate a point across many random trees — shorter path = more anomalous.
# =============================================================================

import math   
import random  


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — ISOLATION TREE NODE
# A single node in one isolation tree.  Each internal node remembers which
# feature was split and the split threshold.  Leaf nodes just store how many
# training samples fell there (used for the path-length correction).
# ─────────────────────────────────────────────────────────────────────────────

class IsolationTreeNode:
    """One node inside an Isolation Tree."""

    def __init__(
        self,
        left=None,          # left child node  (samples < split_value)
        right=None,         # right child node (samples >= split_value)
        split_feature=None, # index of the feature used for the random cut
        split_value=None,   # threshold chosen uniformly at random
        size=0,             # number of training samples that reached this node
        is_leaf=False       # True when we stop splitting (depth or single sample)
    ):
        self.left          = left
        self.right         = right
        self.split_feature = split_feature
        self.split_value   = split_value
        self.size          = size
        self.is_leaf       = is_leaf


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — ISOLATION TREE
# Builds ONE isolation tree on a random subsample, then lets you measure
# the path length (number of edges root → leaf) for any new point.
# ─────────────────────────────────────────────────────────────────────────────

class IsolationTree:
    """A single fully-random binary tree used inside IsolationForest."""

    def __init__(self, max_depth, feature_indices=None):
        # max_depth caps tree growth so we don't recurse forever.
        # In the original paper: max_depth = ceil(log2(subsample_size))
        self.max_depth       = max_depth
        # feature_indices: list of column indices this tree is allowed to split
        # on.  None means use ALL features (default / backward-compatible).
        # Set by IsolationForestScratch when max_features < 1.0 — each tree
        # gets a random subset of features, just like sklearn's implementation.
        self.feature_indices = feature_indices
        self.root            = None   # filled in by fit()

    # ── BUILD THE TREE ────────────────────────────────────────────────────────

    def fit(self, X, depth=0):
        """
        Recursively build an isolation tree.

        Parameters
        ----------
        X     : list of lists — the training rows available at this node
        depth : int           — current recursion depth (starts at 0)

        Returns
        -------
        IsolationTreeNode — the root of the subtree built from X
        """
        n_samples  = len(X)          # how many rows arrived at this node
        n_features = len(X[0])       # number of input features

        # ── STOPPING CONDITIONS ──────────────────────────────────────────────
        # Stop splitting when:
        #   (a) only one sample left (can't split further), OR
        #   (b) we have reached the maximum allowed depth.
        # In both cases create a LEAF and record how many samples are here.
        # The size is used later in the path-length correction formula c(n).
        if n_samples <= 1 or depth >= self.max_depth:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        # ── RANDOM FEATURE SELECTION ──────────────────────────────────────────
        # Pick one feature uniformly at random from the allowed feature set.
        # If feature_indices is set (max_features < 1.0), only those columns
        # are candidates — mirrors sklearn's per-tree feature subsampling.
        # Otherwise fall back to sampling from the full feature range.
        available_features = (
            self.feature_indices                   # use restricted set
            if self.feature_indices is not None
            else list(range(n_features))           # use all features
        )
        feature_idx = random.choice(available_features)

        # ── GET THE VALUE RANGE FOR THAT FEATURE ─────────────────────────────
        # Collect every value in this column so we know [min, max].
        col_values = [row[feature_idx] for row in X]
        min_val    = min(col_values)
        max_val    = max(col_values)

        # ── GUARD: ALL VALUES IDENTICAL ───────────────────────────────────────
        # If every row has the same value for this feature we cannot split.
        # Treat the node as a leaf (all these samples are indistinguishable).
        if min_val == max_val:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        # ── RANDOM SPLIT THRESHOLD ────────────────────────────────────────────
        # Draw the split point uniformly from (min_val, max_val).
        # Points strictly less than the threshold go left; the rest go right.
        split_value = random.uniform(min_val, max_val)

        # ── PARTITION DATA ────────────────────────────────────────────────────
        left_data  = [row for row in X if row[feature_idx] <  split_value]
        right_data = [row for row in X if row[feature_idx] >= split_value]

        # ── BUILD THE INTERNAL NODE ───────────────────────────────────────────
        node               = IsolationTreeNode(size=n_samples, is_leaf=False)
        node.split_feature = feature_idx
        node.split_value   = split_value

        # Recurse: build left and right subtrees, each one level deeper
        node.left  = self.fit(left_data,  depth + 1)
        node.right = self.fit(right_data, depth + 1)

        return node

    # ── PATH LENGTH QUERY ─────────────────────────────────────────────────────

    def path_length(self, x, node, current_depth=0):
        """
        Traverse the tree for sample x and return its path length.

        The path length is the number of edges from the root to the leaf where
        x lands.  At a leaf we add a correction term c(node.size) that accounts
        for the virtual splits that would have continued past the depth limit.

        Parameters
        ----------
        x             : list/array — one data sample
        node          : IsolationTreeNode — current node (start with tree.root)
        current_depth : int — edges traversed so far (starts at 0)

        Returns
        -------
        float — estimated path length for x in this tree
        """
        # ── LEAF: return depth + correction for remaining sub-tree ────────────
        # c(n) estimates the average additional path length if we hadn't stopped.
        if node.is_leaf:
            return current_depth + self._c(node.size)

        # ── INTERNAL NODE: follow the appropriate branch ──────────────────────
        # Same rule that was used during training.
        if x[node.split_feature] < node.split_value:
            return self.path_length(x, node.left,  current_depth + 1)
        else:
            return self.path_length(x, node.right, current_depth + 1)

    # ── HELPER: AVERAGE PATH LENGTH CORRECTION ────────────────────────────────

    @staticmethod
    def _c(n):
        """
        Expected path length for an unsuccessful Binary Search Tree (BST) search
        in a BST built on n randomly ordered keys.

        Formula (from the paper):
            c(n) = 2 * H(n-1) - 2*(n-1)/n
        where H(k) = ln(k) + γ  is the k-th harmonic number approximated via
        the natural log plus the Euler–Mascheroni constant γ ≈ 0.5772156649.

        Special cases:
            c(1) = 0  — single sample, no split needed
            c(2) = 1  — two samples, exactly one split needed
        """
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        # Euler–Mascheroni constant (γ), used to approximate harmonic numbers
        EULER_MASCHERONI = 0.5772156649
        harmonic_n_minus_1 = math.log(n - 1) + EULER_MASCHERONI  # H(n-1)
        return 2.0 * harmonic_n_minus_1 - (2.0 * (n - 1) / n)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ISOLATION FOREST (THE ENSEMBLE)
# Fits many IsolationTrees on random subsamples and combines their path
# lengths into an anomaly score, then thresholds that score by contamination.
# ─────────────────────────────────────────────────────────────────────────────

class IsolationForestScratch:
    """
    Isolation Forest anomaly detector — sklearn-compatible interface,
    implemented in pure Python (no numpy / scipy / sklearn).

    Parameters
    ----------
    n_estimators  : int   — number of isolation trees to build (default 100)
    max_samples   : int   — number of rows subsampled per tree (default 256)
    contamination : float — expected fraction of anomalies in the data,
                            used to set the decision boundary (default 0.1)
    random_state  : int   — seed for reproducibility (default None = random)

    Attributes set after fit()
    --------------------------
    trees_     : list[IsolationTree] — the fitted trees
    threshold_ : float              — anomaly-score cutoff derived from contamination
    n_samples_  : int               — subsample size actually used per tree
    """

    def __init__(
        self,
        n_estimators=100,
        max_samples=256,
        contamination=0.1,
        max_features=1.0,
        random_state=None
    ):
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.contamination = contamination
        # max_features controls how many features each tree is allowed to split
        # on — identical to sklearn's max_features parameter:
        #   float in (0,1] → fraction of total features  e.g. 0.8 = 80%
        #   int ≥ 1        → exact number of features
        #   1.0 (default)  → use all features (original paper behaviour)
        self.max_features  = max_features
        self.random_state  = random_state

        # set after fit
        self.trees_     = []
        self.threshold_ = None
        self.n_samples_ = None   # actual subsample size (<=max_samples)

    # ── FIT ───────────────────────────────────────────────────────────────────

    def fit(self, X):
        """
        Build the forest.

        Steps
        -----
        1. Seed the RNG for reproducibility (if requested).
        2. Convert X to a plain Python list-of-lists (handles numpy arrays).
        3. Decide the actual subsample size and tree depth.
        4. Build n_estimators trees, each on a fresh random subsample.
        5. Score every training point and set the decision threshold so that
           exactly `contamination` fraction are flagged as anomalies.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        # ── SEED RNG ──────────────────────────────────────────────────────────
        if self.random_state is not None:
            random.seed(self.random_state)

        # ── CONVERT INPUT TO PLAIN LIST-OF-LISTS ─────────────────────────────
        # We do this so the code works whether X is a numpy array, a pandas
        # DataFrame, or a plain Python list — all accessed via list indexing.
        X_list = [list(row) for row in X]
        n_total = len(X_list)   # total training rows

        # ── SUBSAMPLE SIZE ────────────────────────────────────────────────────
        # The paper recommends 256 as the default — larger subsamples do not
        # improve accuracy much but slow down training significantly.
        # We also cap it so we never request more samples than we have.
        actual_max_samples = min(self.max_samples, n_total)
        self.n_samples_ = actual_max_samples

        # ── MAX TREE DEPTH ────────────────────────────────────────────────────
        # ceil(log2(subsample_size)) ensures the tree is deep enough to isolate
        # every point in the subsample, yet bounded to prevent overly long paths.
        if actual_max_samples > 1:
            max_depth = math.ceil(math.log2(actual_max_samples))
        else:
            max_depth = 1   # degenerate: only 1 sample, depth is irrelevant

        # ── FEATURE COUNT ─────────────────────────────────────────────────────
        # Resolve max_features to an integer number of columns to use.
        n_features = len(X_list[0]) if X_list else 0
        if isinstance(self.max_features, float):
            # e.g. 0.8 → 80% of columns, at least 1
            n_features_to_use = max(1, int(self.max_features * n_features))
        else:
            # integer given directly — clamp between 1 and n_features
            n_features_to_use = max(1, min(int(self.max_features), n_features))

        # ── BUILD EACH TREE ───────────────────────────────────────────────────
        self.trees_ = []
        for _ in range(self.n_estimators):
            # random.sample draws WITHOUT replacement — important so each
            # subsample is IID and cannot have duplicate rows inflating results.
            subsample = random.sample(X_list, actual_max_samples)

            # Draw a random feature subset for this tree (without replacement).
            # Each tree sees a different random column set — this is the
            # same strategy sklearn uses for IsolationForest max_features.
            if n_features_to_use < n_features:
                feat_idx = random.sample(range(n_features), n_features_to_use)
            else:
                feat_idx = None   # None = use all (avoids unnecessary list copy)

            tree      = IsolationTree(max_depth=max_depth, feature_indices=feat_idx)
            tree.root = tree.fit(subsample)   # build the tree recursively
            self.trees_.append(tree)

        # ── SET DECISION THRESHOLD ────────────────────────────────────────────
        # Score every training sample; sort scores descending (higher = more
        # anomalous) and pick the score that keeps exactly `contamination`
        # fraction of points above it.  This is equivalent to sklearn's
        # behaviour with a fitted contamination.
        train_scores = self._anomaly_scores(X_list)
        sorted_scores = sorted(train_scores, reverse=True)  # descending
        # Index of the last sample that counts as anomalous
        cutoff_idx    = max(0, int(self.contamination * len(sorted_scores)) - 1)
        self.threshold_ = sorted_scores[cutoff_idx]

        return self   # return self so `model = IF().fit(X)` works

    # ── PREDICT ───────────────────────────────────────────────────────────────

    def predict(self, X):
        """
        Classify each sample as normal (+1) or anomaly (-1).

        Mirrors sklearn's IsolationForest.predict() convention exactly:
            +1  →  normal (anomaly score < threshold)
            -1  →  anomaly (anomaly score >= threshold)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        list[int] — length n_samples, values in {-1, +1}
        """
        X_list = [list(row) for row in X]       # convert to plain lists
        scores  = self._anomaly_scores(X_list)  # compute anomaly scores

        # -1 = anomaly when score is at or above the fitted threshold
        # +1 = normal  when score is below the threshold
        predictions = [-1 if s >= self.threshold_ else 1 for s in scores]
        return predictions

    # ── DECISION FUNCTION ─────────────────────────────────────────────────────

    def decision_function(self, X):
        """
        Return raw anomaly scores for each sample.

        Score interpretation:
            score ≈ 1.0  → very anomalous (isolated very quickly)
            score ≈ 0.5  → indeterminate
            score ≈ 0.0  → very normal (deep in a dense region)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        list[float] — anomaly scores in (0, 1)
        """
        X_list = [list(row) for row in X]
        return self._anomaly_scores(X_list)

    # ── INTERNAL: COMPUTE ANOMALY SCORES ─────────────────────────────────────

    def _anomaly_scores(self, X_list):
        """
        Compute anomaly score for every row in X_list.

        Formula (from the paper):
            s(x, n) = 2 ^ ( -E[h(x)] / c(n) )

        where:
            E[h(x)] = average path length across all trees
            c(n)    = expected path length for n samples (normalisation factor)

        The normalisation by c(n) ensures scores are comparable across
        datasets of different sizes.

        Returns
        -------
        list[float] — one score per row, values in (0, 1)
        """
        # c(n) normalisation constant, computed once for the subsample size
        c_n = IsolationTree._c(self.n_samples_)

        # Avoid division by zero when n_samples_ == 1
        if c_n == 0:
            return [0.5] * len(X_list)

        scores = []
        for x in X_list:
            # Average path length across all trees for this one sample
            avg_path = sum(
                tree.path_length(x, tree.root)
                for tree in self.trees_
            ) / len(self.trees_)

            # Anomaly score: large avg_path → score near 0 (normal)
            #                small avg_path → score near 1 (anomalous)
            score = 2.0 ** (-avg_path / c_n)
            scores.append(score)

        return scores


# =============================================================================
# SECTION 4 — EVALUATION METRICS (ALL FROM SCRATCH)
# These replicate sklearn.metrics: confusion_matrix, accuracy_score,
# and classification_report — working on binary 0/1 integer labels.
# =============================================================================

def confusion_matrix_scratch(y_true, y_pred):
    """
    Compute a 2x2 confusion matrix for binary classification.

    Layout (same as sklearn):
        [[TN, FP],
         [FN, TP]]

    Where:
        TN — True Negative:  both true and predicted are 0 (normal)
        FP — False Positive: true is 0 (normal) but predicted is 1 (anomaly)
        FN — False Negative: true is 1 (anomaly) but predicted is 0 (normal)
        TP — True Positive:  both true and predicted are 1 (anomaly)

    Parameters
    ----------
    y_true : list/array of int — ground-truth labels (0 = normal, 1 = anomaly)
    y_pred : list/array of int — predicted labels    (0 = normal, 1 = anomaly)

    Returns
    -------
    list[list[int]] — [[TN, FP], [FN, TP]]
    """
    # Count each of the four combinations with a single pass
    TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    TN = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)  # false alarm
    FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)  # missed anomaly

    return [[TN, FP],
            [FN, TP]]


def accuracy_score_scratch(y_true, y_pred):
    """
    Fraction of samples for which the prediction matches the true label.

        accuracy = (number of correct predictions) / (total predictions)

    Parameters
    ----------
    y_true : list/array of int
    y_pred : list/array of int

    Returns
    -------
    float — accuracy in [0, 1]
    """
    if len(y_true) == 0:
        return 0.0  # guard against empty input

    # Count how many positions have matching values
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def classification_report_scratch(y_true, y_pred, labels=None):
    """
    Per-class precision, recall, F1-score and support — plus accuracy,
    macro-average and weighted-average rows.

    Definitions
    -----------
    Precision  (P) = TP / (TP + FP)     — of all predicted positives, how many are correct?
    Recall     (R) = TP / (TP + FN)     — of all true positives, how many did we catch?
    F1-score       = 2*P*R / (P+R)      — harmonic mean of precision and recall
    Support        = number of true samples in that class

    Macro average  — simple average of per-class metrics (treats each class equally)
    Weighted avg   — per-class metrics weighted by class support

    Parameters
    ----------
    y_true  : list/array of int — ground-truth labels
    y_pred  : list/array of int — predicted labels
    labels  : list of int, optional — classes to include; defaults to all seen in y_true

    Returns
    -------
    str — formatted report string (print it)
    """
    # Determine which classes to report on
    if labels is None:
        classes = sorted(set(y_true))   # all unique classes, sorted
    else:
        classes = sorted(labels)

    # ── PER-CLASS STATISTICS ──────────────────────────────────────────────────
    per_class = {}
    for cls in classes:
        # True Positive: predicted cls AND actually cls
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        # False Positive: predicted cls BUT actually something else
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        # False Negative: actually cls BUT predicted something else
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        # Support: total samples with true label == cls
        support = sum(1 for t in y_true if t == cls)

        # Precision: 0.0 when denominator is zero (no predictions for this class)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Recall: 0.0 when there are no true positives in this class at all
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # F1: harmonic mean — 0.0 when both precision and recall are 0
        f1        = (2.0 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[cls] = {
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'support':   support
        }

    # ── OVERALL ACCURACY ──────────────────────────────────────────────────────
    total   = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / total if total > 0 else 0.0

    # ── MACRO AVERAGE ─────────────────────────────────────────────────────────
    # Simple unweighted mean of per-class metrics
    n_classes     = len(classes)
    macro_p  = sum(per_class[c]['precision'] for c in classes) / n_classes if n_classes else 0.0
    macro_r  = sum(per_class[c]['recall']    for c in classes) / n_classes if n_classes else 0.0
    macro_f1 = sum(per_class[c]['f1']        for c in classes) / n_classes if n_classes else 0.0

    # ── WEIGHTED AVERAGE ──────────────────────────────────────────────────────
    # Weight each class's metric by its support
    total_support = sum(per_class[c]['support'] for c in classes)
    if total_support > 0:
        w_p  = sum(per_class[c]['precision'] * per_class[c]['support'] for c in classes) / total_support
        w_r  = sum(per_class[c]['recall']    * per_class[c]['support'] for c in classes) / total_support
        w_f1 = sum(per_class[c]['f1']        * per_class[c]['support'] for c in classes) / total_support
    else:
        w_p = w_r = w_f1 = 0.0

    # ── FORMAT OUTPUT ─────────────────────────────────────────────────────────
    col_w = 11   # column width for numeric fields
    header = f"{'':>12}  {'precision':>{col_w}}  {'recall':>{col_w}}  {'f1-score':>{col_w}}  {'support':>{col_w}}"
    sep    = ""

    lines = [header, sep]

    for cls in classes:
        s = per_class[cls]
        label = str(cls)
        lines.append(
            f"{label:>12}  {s['precision']:>{col_w}.4f}  {s['recall']:>{col_w}.4f}"
            f"  {s['f1']:>{col_w}.4f}  {s['support']:>{col_w}}"
        )

    lines.append(sep)
    lines.append(
        f"{'accuracy':>12}  {'':>{col_w}}  {'':>{col_w}}"
        f"  {accuracy:>{col_w}.4f}  {total:>{col_w}}"
    )
    lines.append(
        f"{'macro avg':>12}  {macro_p:>{col_w}.4f}  {macro_r:>{col_w}.4f}"
        f"  {macro_f1:>{col_w}.4f}  {total_support:>{col_w}}"
    )
    lines.append(
        f"{'weighted avg':>12}  {w_p:>{col_w}.4f}  {w_r:>{col_w}.4f}"
        f"  {w_f1:>{col_w}.4f}  {total_support:>{col_w}}"
    )

    return "\n".join(lines)
