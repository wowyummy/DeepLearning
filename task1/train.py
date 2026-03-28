import time
import numpy as np

from model import NeuralNetwork,FullyConnectedLayer
from utils import create_mini_batches, accuracy_score


def evaluate_model(model, X, y_true, y_one_hot, training=False):
    """
    评估：返回损失与准确率。
    """
    probs = model.forward(X, training=training)
    loss = model.compute_loss(y_one_hot)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    return loss, acc, y_pred


def train_one_experiment(config, data):
    """
    训练单个实验配置。
    config 字段示例：
    {
        "name": "Baseline",
        "hidden_dims": [256, 128],
        "activation": "relu",
        "learning_rate": 0.01,
        "batch_size": 64,
        "l2_lambda": 1e-4,
        "epochs": 15,
        "optimizer": "sgd"
    }
    """
    X_train = data["X_train"]
    y_train = data["y_train"]
    y_train_one_hot = data["y_train_one_hot"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    y_val_one_hot = data["y_val_one_hot"]

    model = NeuralNetwork(
        input_dim=784,
        hidden_dims=config["hidden_dims"],
        output_dim=10,
        hidden_activation=config["activation"],
        l2_lambda=config["l2_lambda"],
        init_method=config.get("init_method", "auto"),
        dropout_rate=config.get("dropout_rate", 0.0),
        seed=config.get("seed", 42),
    )

    #初始化训练历史记录字典，用于存储每个 epoch 的训练和验证损失与准确率
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = -1.0 
    # 训练过程中保存验证集上表现最好的模型参数
    best_state = None     
    # 全局迭代步数（累计处理的mini-batch数量），用于Adam等需要时间步的优化器。
    t_global = 0    
    start_time = time.time()

    for epoch in range(1, config["epochs"] + 1):
        # ---- 训练阶段 ----
        for X_batch, y_batch in create_mini_batches(  # 逐批读取数据
            X_train,
            y_train_one_hot,
            batch_size=config["batch_size"],
            shuffle=True,
            seed=config.get("seed", 42) + epoch,
        ):
            _ = model.forward(X_batch, training=True) #_表示忽略返回值, 前向传播得到输出概率，实际训练过程中不需要使用这个返回值，因为损失和梯度计算会直接使用模型内部缓存的 logits 和 probs。
            model.backward(y_batch)

            t_global += 1
            model.update(
                learning_rate=config["learning_rate"],
                optimizer=config.get("optimizer", "sgd"),
                t=t_global,
                beta=config.get("beta", 0.9),
                beta1=config.get("beta1", 0.9),
                beta2=config.get("beta2", 0.999),
            )

        # ---- 每个 epoch 结束后做一次完整评估 ----
        train_loss, train_acc, _ = evaluate_model(model, X_train, y_train, y_train_one_hot, training=False)
        val_loss, val_acc, _ = evaluate_model(model, X_val, y_val, y_val_one_hot, training=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最优参数副本
            best_state = []
            for layer in model.layers:
                if isinstance(layer, FullyConnectedLayer):
                    best_state.append({
                        'W': layer.W.copy(),
                        'b': layer.b.copy(),
                        'vW': layer.vW.copy(),
                        'vb': layer.vb.copy(),
                        'mW': layer.mW.copy(),
                        'mb': layer.mb.copy(),
                        'sW': layer.sW.copy(),
                        'sb': layer.sb.copy(),
                    })
                else:
                    best_state.append(None)
        print(
            f"[{config['name']}] Epoch {epoch:02d}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # 恢复最优验证集参数
    if best_state is not None:
        for layer, state in zip(model.layers, best_state):
            if isinstance(layer, FullyConnectedLayer) and state is not None:
                layer.W = state['W']
                layer.b = state['b']
                layer.vW = state['vW']
                layer.vb = state['vb']
                layer.mW = state['mW']
                layer.mb = state['mb']
                layer.sW = state['sW']
                layer.sb = state['sb']

    # 计算训练耗时
    elapsed = time.time() - start_time
    result = {
        "config": config,
        "model": model,
        "history": history,
        "best_val_acc": best_val_acc,
        "train_time_sec": elapsed,
    }
    return result


def run_all_experiments(experiment_configs, data):
    """
    依次运行多组超参数实验，并返回结果列表。
    """
    results = []
    for cfg in experiment_configs:
        print("=" * 80) 
        print(f"开始实验: {cfg['name']}")
        print(cfg)
        result = train_one_experiment(cfg, data)
        results.append(result)
    return results
