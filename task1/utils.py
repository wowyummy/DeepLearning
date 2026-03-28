import os
import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


def one_hot_encode(y, num_classes=10):
    """
    将标签向量编码为 one-hot 矩阵
    输入 y 是一个形状为 (m,) 的整数数组,包含了每个样本的类别标签(0-9)。
    """
    y = y.astype(int)
    # 创建一个全零的矩阵，行数等于样本数，列数等于类别数。
    one_hot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    # 使用高级索引将每个样本对应的标签位置批量设置为 1.0。
    one_hot[np.arange(y.shape[0]), y] = 1.0
    return one_hot


def train_val_test_split_mnist(X, y, val_ratio=0.2, seed=42):
    """
    数据划分：
    1. 样本数为 70000(标准 MNIST):前 60000 作为 train_val,后 10000 作为 test。
    2. 再将 train_val 按 8:2 划分 train/val。
    3. X: (m, 784) 的特征矩阵,y: (m,) 的标签向量。
    """
    rng = np.random.default_rng(seed) # 可复现的随机数生成器

    if X.shape[0] >= 70000:
        X_train_val = X[:60000]
        y_train_val = y[:60000]
        X_test = X[60000:70000]
        y_test = y[60000:70000]
    else:
        # 如果不是标准 70000 数据，按 8:2 划出 test
        indices = np.arange(X.shape[0])
        rng.shuffle(indices) # 打乱索引以随机划分数据
        split_test = int(X.shape[0] * 0.8)
        train_val_idx = indices[:split_test]
        test_idx = indices[split_test:]
        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

    # 再将 train_val 按 8:2 划分 train/val
    indices = np.arange(X_train_val.shape[0])
    rng.shuffle(indices)
    split_val = int(X_train_val.shape[0] * (1 - val_ratio))

    train_idx = indices[:split_val]
    val_idx = indices[split_val:]

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_and_preprocess_mnist(seed=42):
    """
    加载并预处理 MNIST:
    1. 从 OpenML 下载
    2. 像素值归一化到 [0,1]
    3. 标签转 int
    4. 划分 train/val/test
    5. 生成 one-hot 标签
    """
    print("正在从 OpenML 加载 MNIST")
    """
    fetch_openml:sklearn 提供的从 OpenML 加载数据集的函数：
    - name: 数据集名称，这里是 "mnist_784",784 是指每个样本有 784 个特征(28x28 图像展平）。
    - version: 数据集版本
    - as_frame: 是否以 DataFrame 格式返回数据和标签，这里设置为 False 以获得 numpy 数组。
    - mnist: 包含数据和标签的 Bunch 对象。
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False,data_home="./my_data")

    # 数据预处理
    # mnist.data 是一个 (70000, 784) 的数组，每个元素是 0-255 的整数（对应像素灰度值）
    # mnist.target 是一个 (70000,) 的数组，包含了每个样本的标签（0-9）。
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    # 划分数据集并生成 one-hot 标签
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split_mnist(X, y, val_ratio=0.2, seed=seed)

    y_train_one_hot = one_hot_encode(y_train, num_classes=10)
    y_val_one_hot = one_hot_encode(y_val, num_classes=10)
    y_test_one_hot = one_hot_encode(y_test, num_classes=10)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "y_train_one_hot": y_train_one_hot,
        "X_val": X_val,
        "y_val": y_val,
        "y_val_one_hot": y_val_one_hot,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_one_hot": y_test_one_hot,
    }


def create_mini_batches(X, y_one_hot, batch_size, shuffle=True, seed=None):
    """
    Mini-batch 生成器。
    每次返回 (X_batch, y_batch)。
    """
    m = X.shape[0]
    indices = np.arange(m) # 0 到 m-1 的整数数组，用于索引原始数据

    if shuffle:
        if seed is None:
            np.random.shuffle(indices)
        else:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

    for start in range(0, m, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        # yield 关键字使函数成为一个生成器，每次调用 next() 时返回一个批次的数据 (X_batch, y_batch)，并在下一次调用时继续执行函数直到下一个 yield。
        yield X[batch_idx], y_one_hot[batch_idx]


def accuracy_score(y_true, y_pred):
    """
    计算准确率：正确预测的样本数 / 总样本数。
    """
    return np.mean(y_true == y_pred)


def confusion_matrix_np(y_true, y_pred, num_classes=10):
    """
    计算混淆矩阵。
    行 = 真实标签，列 = 预测标签，值 = 对应样本数
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    # zip(y_true, y_pred) 会同时迭代 y_true 和 y_pred 中的元素，每次返回一对 (t, p)，其中 t 是真实标签，p 是预测标签。
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_loss_acc_curves(history, title="Training Curves", save_path=None):
    """
    绘制训练/验证损失与准确率曲线。
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title} - Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    """
    绘制混淆矩阵。
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    # 在矩阵格子中标数字
    thresh = cm.max() * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_misclassified_samples(X, y_true, y_pred, max_show=16, save_path=None):
    """
    可视化分类错误样本
    """
    wrong_idx = np.where(y_true != y_pred)[0]

    if wrong_idx.size == 0:
        print("没有分类错误样本可展示。")
        return

    show_n = min(max_show, wrong_idx.size)
    idx = wrong_idx[:show_n]

    cols = 4
    rows = int(np.ceil(show_n / cols))

    plt.figure(figsize=(10, 2.5 * rows))
    for i, sample_idx in enumerate(idx):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X[sample_idx].reshape(28, 28), cmap="gray")
        plt.title(f"T:{y_true[sample_idx]} P:{y_pred[sample_idx]}")
        plt.axis("off")

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
