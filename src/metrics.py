"""評估指標。

- beat(mitdb)：AAMI 每類 Se/PPV/F1 + macro-F1（盯 S/V）。
- AF(afdb)：AF-class segment F1 + per-record accuracy（不只報 overall）。
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def macro_f1_score(y_true, y_pred, num_classes):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    return float(np.mean(f1))


def aami_report(y_true, y_pred, class_names, focus=("S", "V")):
    labels = list(range(len(class_names)))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = [
        {
            "class": class_names[i],
            "sensitivity": float(rec[i]),
            "ppv": float(prec[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
        }
        for i in range(len(class_names))
    ]
    return {
        "per_class": per_class,
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": cm,
        "class_names": list(class_names),
        "focus": list(focus),
    }


def format_report(rep):
    lines = []
    lines.append(f"{'class':<8}{'Se(recall)':>12}{'PPV(prec)':>12}{'F1':>10}{'support':>10}")
    lines.append("-" * 52)
    for row in rep["per_class"]:
        lines.append(
            f"{row['class']:<8}{row['sensitivity']:>12.4f}"
            f"{row['ppv']:>12.4f}{row['f1']:>10.4f}{row['support']:>10d}"
        )
    lines.append("-" * 52)
    lines.append(f"macro-F1: {rep['macro_f1']:.4f}")

    focus_rows = [r for r in rep["per_class"] if r["class"] in rep["focus"]]
    if focus_rows:
        foc = "   ".join(f"{r['class']} F1={r['f1']:.4f}" for r in focus_rows)
        lines.append(f"重點類別（S/V）：{foc}")

    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred): " + " ".join(rep["class_names"]))
    cm = rep["confusion_matrix"]
    for i, name in enumerate(rep["class_names"]):
        lines.append(f"  {name}: " + " ".join(f"{v:7d}" for v in cm[i]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AF (afdb) 指標：AF = class 1
# ---------------------------------------------------------------------------
def af_f1(y_true, y_pred):
    """AF-class（正類=1）的 segment F1，用於模型選擇。"""
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def af_report(y_true, y_pred, groups=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rep = {
        "af_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "af_sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "af_ppv": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "af_specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "accuracy": float((y_pred == y_true).mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "per_record": [],
    }
    if groups is not None:
        groups = np.asarray(groups)
        for r in sorted(set(groups.tolist())):
            m = groups == r
            rep["per_record"].append(
                {
                    "record": r,
                    "acc": float((y_pred[m] == y_true[m]).mean()),
                    "n": int(m.sum()),
                    "af_frac": float((y_true[m] == 1).mean()),
                }
            )
    accs = [p["acc"] for p in rep["per_record"]]
    rep["mean_record_acc"] = float(np.mean(accs)) if accs else float("nan")
    return rep


def format_af_report(rep):
    lines = []
    lines.append(
        f"AF-class  F1={rep['af_f1']:.4f}  Se(recall)={rep['af_sensitivity']:.4f}  "
        f"PPV={rep['af_ppv']:.4f}  Specificity={rep['af_specificity']:.4f}"
    )
    lines.append(
        f"overall accuracy={rep['accuracy']:.4f}   |   per-record mean acc={rep['mean_record_acc']:.4f}"
    )
    lines.append("")
    lines.append("Confusion matrix (rows=true, cols=pred):  non-AF  AF")
    cm = rep["confusion_matrix"]
    for i, name in enumerate(["non-AF", "AF"]):
        lines.append(f"  {name:>6}: " + " ".join(f"{v:7d}" for v in cm[i]))
    if rep["per_record"]:
        lines.append("")
        lines.append("Per-record accuracy:")
        for p in rep["per_record"]:
            lines.append(
                f"  {p['record']}: acc={p['acc']:.4f}  n={p['n']:>5d}  AF比例={p['af_frac']:.1%}"
            )
    return "\n".join(lines)
