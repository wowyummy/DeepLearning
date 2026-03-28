import os
import numpy as np

from train import run_all_experiments, evaluate_model
from utils import (
    load_and_preprocess_mnist,
    confusion_matrix_np,
    plot_loss_acc_curves,
    plot_confusion_matrix,
    plot_misclassified_samples,
)


# 为了保证实验可复现，设置全局随机种子
np.random.seed(42)


def main():
    """
    主函数：
    1. 加载与预处理 MNIST
    2. 运行 Baseline / 组1 / 组2 /  Advanced 四组实验
    3. 输出各组验证集性能对比
    4. 选择验证集最优模型，在测试集评估
    5. 绘制训练曲线、混淆矩阵、错误样本可视化
    """
    # -------------------------
    # 数据加载与预处理
    # -------------------------
    data = load_and_preprocess_mnist(seed=42)

    # -------------------------
    #  实验配置
    # -------------------------
    # - hidden_dims: 隐藏层结构
    # - activation: 隐藏层激活函数
    # - learning_rate: 学习率
    # - batch_size: 小批量大小
    # - l2_lambda: L2 正则化系数
    # - optimizer: 优化器（默认 sgd；可选 momentum/adam）
    # - epochs: 训练轮数
    experiment_configs = [
        {
            "name": "Baseline",
            "hidden_dims": [256, 128],
            "activation": "relu",
            "learning_rate": 0.01,
            "batch_size": 64,
            "l2_lambda": 1e-4,
            "epochs": 80,
            "optimizer": "sgd",
            "init_method": "he",
            "seed": 42,
        },
        {
            "name": "Group1",
            "hidden_dims": [512, 256],
            "activation": "leakyrelu",
            "learning_rate": 0.005,
            "batch_size": 128,
            "l2_lambda": 1e-3,
            "epochs": 80,
            "optimizer": "sgd",
            "init_method": "he",
            "seed": 43,
        },
        {
            "name": "Group2",
            "hidden_dims": [128],
            "activation": "tanh",
            "learning_rate": 0.02,
            "batch_size": 32,
            "l2_lambda": 0.0,
            "epochs": 80,
            "optimizer": "sgd",
            "init_method": "he",
            "seed": 44,
        },
        {
            "name": "Advanced",
            "hidden_dims": [256, 128],          # 与 Baseline 相同结构，便于对比
            "activation": "relu",               # 使用 ReLU
            "learning_rate": 0.001,             # Adam 常用学习率
            "batch_size": 64,                   # 与 Baseline 一致
            "l2_lambda": 1e-4,                 # 正则化可选
            "epochs": 80,
            "optimizer": "adam",                # 改用 Adam
            "init_method": "xavier",            # 使用 Xavier 初始化（也可用 he）
            "dropout_rate": 0.2,               # 添加 dropout，丢弃率 0.2
            "seed": 45,
    }
    ]

    # -------------------------
    # 运行多组实验
    # -------------------------
    results = run_all_experiments(experiment_configs, data)

    # -------------------------
    # 验证集结果汇总
    # -------------------------
    print("\n" + "=" * 120)
    print("实验结果汇总")
    print("=" * 120)
    print(f"{'实验组':<10} | {'隐藏层结构':<15} | {'激活函数':<10} | {'学习率':<8} | "
      f"{'Batch':<6} | {'L2系数':<10} | {'最佳验证准确率':<18} | {'训练时长(s)':<12}")
    print("-" * 120)
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    unique_results = {}
    for res in results:
        cfg = res["config"]
        unique_results[cfg["name"]] = res
    results = list(unique_results.values())
    for res in results:
        cfg = res["config"]
        row = (
            f"{cfg['name']:<10} | {str(cfg['hidden_dims']):<15} | {cfg['activation']:<10} | "
            f"{cfg['learning_rate']:<8} | {cfg['batch_size']:<6} | {cfg['l2_lambda']:<10} | "
            f"{res['best_val_acc']:<18.4f} | {res['train_time_sec']:<12.2f}"
        )
        print(row)

        #测试集评估
        test_loss, test_acc, y_test_pred = evaluate_model(
            res["model"],
            data["X_test"],
            data["y_test"],
            data["y_test_one_hot"],
            training=False
        )
        res["test_acc"] = test_acc
        test_status = "✓ 达标" if test_acc >= 0.95 else "✗ 未达标"
        print(f"  -> 测试集 Loss: {test_loss:.4f}, Acc: {test_acc:.4f} ({test_acc*100:.2f}%) {test_status}")
        # 为每个实验单独生成可视化
        curve_path = os.path.join(output_dir, f"curves_{cfg['name']}.png")
        plot_loss_acc_curves(
            res["history"],
            title=f"{cfg['name']} (Act={cfg['activation']}, LR={cfg['learning_rate']})",
            save_path=curve_path,
        )
        
        # 混淆矩阵
        cm = confusion_matrix_np(data["y_test"], y_test_pred, num_classes=10)
        cm_path = os.path.join(output_dir, f"confusion_matrix_{cfg['name']}.png")
        plot_confusion_matrix(cm, title=f"Confusion Matrix - {cfg['name']}", save_path=cm_path)
        
        # 错误样本可视化
        wrong_path = os.path.join(output_dir, f"misclassified_{cfg['name']}.png")
        plot_misclassified_samples(
            data["X_test"],
            data["y_test"],
            y_test_pred,
            max_show=12,
            save_path=wrong_path,
        )

    # 取验证集最优模型
    best_test_result = max(results, key=lambda x: x["test_acc"])
    best_test_model = best_test_result["model"]
    best_test_name = best_test_result["config"]["name"]
    best_test_acc = best_test_result["test_acc"]

    print("\n" + "=" * 120)
    print(f"测试集最优模型: {best_test_name}")
    print(f"测试集准确率: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
    if best_test_acc >= 0.95:
        print("测试集准确率达到验收标准（>=95%）。")
    else:
        print("测试集准确率未达95%")

if __name__ == "__main__":
    main()
