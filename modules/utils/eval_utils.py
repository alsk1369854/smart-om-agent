from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from .. import types


def evaluate_binary_classifier(
    *, 
    preds: list[int], 
    labels: list[int], 
    pos_label: int = 1,
    cm_labels: list[int] = [1, 0] 
) -> types.EvaluationResult:
    return types.EvaluationResult(
        accuracy=float(accuracy_score(labels, preds)),
        f1=float(f1_score(labels, preds, average="binary", pos_label=pos_label, zero_division=1)),
        precision=float(precision_score(labels, preds, average="binary", pos_label=pos_label, zero_division=1)),
        recall=float(recall_score(labels, preds, average="binary", pos_label=pos_label, zero_division=1)),
        cm=confusion_matrix(labels, preds, labels=cm_labels).tolist(),
    )


def print_evaluation_result(*, result: types.EvaluationResult, cm_labels: list[str] = ["1", "0"]) -> None:
    print("=" * 56)
    print("Evaluation Metrics".center(56))
    print("-" * 56)
    print(f"{'Accuracy'.ljust(16)}: {result.accuracy:.4f}")
    print(f"{'F1 Score'.ljust(16)}: {result.f1:.4f}")
    print(f"{'Precision'.ljust(16)}: {result.precision:.4f}")
    print(f"{'Recall'.ljust(16)}: {result.recall:.4f}")
    print("-" * 56)
    print("Confusion Matrix".center(56))
    print("-" * 56)

    # 標題
    header = f"{'':18}{'Pred ' + cm_labels[0]:^14}{'Pred ' + cm_labels[1]:^14}"
    print(header)

    # 內容
    print(f"{'Actual ' + cm_labels[0]:18}{str(result.cm[0][0]):^14}{str(result.cm[0][1]):^14}")
    print(f"{'Actual ' + cm_labels[1]:18}{str(result.cm[1][0]):^14}{str(result.cm[1][1]):^14}")

    print("=" * 56)