import numpy as np

class DropoutLayer:
    """
    Dropout 层：以概率 dropout_rate 随机将神经元输出置为 0,
    并缩放剩余激活值，保持期望不变。
    """
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None          # 训练时使用的掩码
        self.cache = None         # 缓存输入，用于反向传播

    def forward(self, X, training=True):
        """
        前向传播：
        - 训练模式：生成掩码，按 dropout_rate 丢弃神经元，并缩放
        - 测试模式：直接返回输入（不进行 dropout)
        """
        if training and self.dropout_rate > 0:
            # 保留概率 p = 1 - dropout_rate
            p = 1.0 - self.dropout_rate
            # 生成伯努利掩码（元素为 0 或 1），保留概率为 p
            self.mask = (np.random.rand(*X.shape) < p).astype(float)
            # 缩放输出，使期望值不变
            out = X * self.mask / p
        else:
            out = X
        self.cache = X
        return out

    def backward(self, dA):
        """
        反向传播：将梯度通过掩码传播，并缩放。
        """
        if self.mask is not None:
            p = 1.0 - self.dropout_rate
            return dA * self.mask / p
        else:
            return dA
        
class FullyConnectedLayer:
    """
    全连接层（线性层 + 可选激活函数）
    线性计算:Z=X⋅W+b
    激活输出:A=g(Z)(g 是激活函数）
    1. 本层维护参数 W、b,以及反向传播得到的梯度 dW、db。
    2. 支持激活函数:relu / sigmoid / tanh / leakyrelu / linear。
    3. forward 会缓存输入 X 和线性输出 Z,便于 backward 使用。
    4.反向传播:通过输出梯度反推参数梯度(∂Loss​/∂W、∂Loss​/∂b)和输入梯度(∂Loss​/∂X)
    """
    # 初始化参数 W、b 和反向传播缓存
    def __init__(self, input_dim, output_dim, activation="relu", init_method="auto", seed=None):
        input_dim = int(input_dim)
        output_dim = int(output_dim)
        if seed is not None:
            # 仅对当前层设置随机种子，便于复现实验
            rng = np.random.default_rng(seed)
            randn = rng.standard_normal
        else:
            randn = np.random.randn

        #activation以小写形式存储
        self.activation = activation.lower()

        # 初始化策略：选择参数初始化缩放系数
        # - ReLU / LeakyReLU 常用 He 初始化 缩放系数√(2/input_dim)
        # - Sigmoid / Tanh 常用 Xavier 初始化 缩放系数√(1/input_dim)
        # - linear 使用较小随机值
        if init_method == "auto":
            if self.activation in ["relu", "leakyrelu"]:
                scale = np.sqrt(2.0 / input_dim)  # He
            elif self.activation in ["sigmoid", "tanh"]:
                scale = np.sqrt(1.0 / input_dim)  # Xavier
            else:
                scale = 0.01  # 线性激活用小随机数
        elif init_method == "he":
            scale = np.sqrt(2.0 / input_dim)
        elif init_method == "xavier":
            scale = np.sqrt(1.0 / input_dim)
        else:
            scale = 0.01

        # 使用缩放后的随机数初始化权重 W，偏置 b 初始化为零。
        self.W = randn((input_dim, output_dim)) * scale
        self.b = np.zeros((1, output_dim))

        # 反向传播缓存
        self.X_cache = None
        self.Z_cache = None

        # 梯度缓存
        self.dW = None
        self.db = None

        # 优化器状态缓存,动量与 Adam 所需状态
        # Momentum vW、vb
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
        # Adam 一阶矩（均值）、二阶矩（方差）
        self.mW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.sW = np.zeros_like(self.W)
        self.sb = np.zeros_like(self.b)

    # 激活函数及其导数
    def _activate(self, Z):
        if self.activation == "relu":
            return np.maximum(0, Z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -50, 50)))
        if self.activation == "tanh":
            return np.tanh(Z)
        if self.activation == "leakyrelu":
            return np.where(Z > 0, Z, 0.01 * Z)
        if self.activation == "linear":
            return Z
        raise ValueError(f"不支持的激活函数: {self.activation}")
    
    # 激活函数的导数，用于反向传播计算 dZ。
    def _activation_grad(self, Z):
        if self.activation == "relu":
            return (Z > 0).astype(float)
        if self.activation == "sigmoid":
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -50, 50)))
            return A * (1.0 - A)
        if self.activation == "tanh":
            A = np.tanh(Z)
            return 1.0 - A * A
        if self.activation == "leakyrelu":
            grad = np.ones_like(Z)
            grad[Z <= 0] = 0.01
            return grad
        if self.activation == "linear":
            return np.ones_like(Z)
        raise ValueError(f"不支持的激活函数: {self.activation}")

    def forward(self, X):
        """
        前向传播:A = activation(XW + b)
        """
        self.X_cache = X
        Z = X @ self.W + self.b  # @ 运算符表示矩阵乘法
        self.Z_cache = Z # 缓存线性输出 Z 以供反向传播使用
        A = self._activate(Z)
        return A

    def backward(self, dA, l2_lambda=0.0):
        """
        反向传播。
        - dA: 损失对当前层激活输出 A 的梯度 ∂Loss/∂A
        - dX: 输入梯度
        """
        X = self.X_cache
        Z = self.Z_cache
        m = X.shape[0]

        dZ = dA * self._activation_grad(Z)

        # 梯度包含 L2 正则项：lambda/m * W
        self.dW = (X.T @ dZ) / m + (l2_lambda / m) * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T
        return dX

    def backward_from_dZ(self, dZ, l2_lambda=0.0):
        """
        当外部已直接得到损失对当前层线性输出 dZ( softmax + 交叉熵组合）时，
        可直接调用本函数，避免重复乘激活导数。
        """
        X = self.X_cache
        m = X.shape[0]

        self.dW = (X.T @ dZ) / m + (l2_lambda / m) * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dX = dZ @ self.W.T
        return dX

    def update_parameters(
        self,
        learning_rate,
        optimizer="sgd",
        beta=0.9,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        t=1,
    ):
        """
        参数更新：支持 SGD / Momentum / Adam。
        """
        opt = optimizer.lower()

        if opt == "sgd":
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db
            return

        if opt == "momentum":
            self.vW = beta * self.vW + (1 - beta) * self.dW
            self.vb = beta * self.vb + (1 - beta) * self.db
            self.W -= learning_rate * self.vW
            self.b -= learning_rate * self.vb
            return

        if opt == "adam":
            # 一阶矩估计
            self.mW = beta1 * self.mW + (1 - beta1) * self.dW
            self.mb = beta1 * self.mb + (1 - beta1) * self.db
            # 二阶矩估计
            self.sW = beta2 * self.sW + (1 - beta2) * (self.dW ** 2)
            self.sb = beta2 * self.sb + (1 - beta2) * (self.db ** 2)

            # 偏差修正
            mW_hat = self.mW / (1 - beta1 ** t)
            mb_hat = self.mb / (1 - beta1 ** t)
            sW_hat = self.sW / (1 - beta2 ** t)
            sb_hat = self.sb / (1 - beta2 ** t)

            self.W -= learning_rate * mW_hat / (np.sqrt(sW_hat) + epsilon)
            self.b -= learning_rate * mb_hat / (np.sqrt(sb_hat) + epsilon)
            return

        raise ValueError(f"不支持的优化器: {optimizer}")


class NeuralNetwork:
    """
    多层全连接神经网络：
    Input -> (FC+激活)*N -> FC(linear) -> Softmax
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        hidden_activation="relu",
        l2_lambda=0.0,
        init_method="auto",
        dropout_rate=0.0,
        seed=42,
    ):
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.layers = []

        dims = [input_dim] + hidden_dims + [output_dim]

        # 循环创建隐藏层
        for i in range(len(hidden_dims)):
            self.layers.append(
                FullyConnectedLayer(
                    dims[i], # 输入维度
                    dims[i + 1], # 输出维度
                    activation=hidden_activation,
                    init_method=init_method,
                    seed=seed + i, # 每层使用不同的随机种子，保证参数初始化的多样性
                )
            )
            # 添加 Dropout 层（如果 dropout_rate > 0 且不是最后一层隐藏层）
            if dropout_rate > 0 and i < len(hidden_dims) - 1:
                self.layers.append(DropoutLayer(dropout_rate))

        # 输出层
        self.layers.append(
            FullyConnectedLayer(
                dims[-2], # 最后一个隐藏层的输出维度
                dims[-1], # output_dim
                activation="linear",
                init_method=init_method,
                seed=seed + len(hidden_dims),
            )
        )

        # 前向传播缓存
        self.logits = None
        self.probs = None

    # 静态方法，不需要实例化即可调用，也不接收 self 参数，主要用于与类逻辑相关但不依赖实例状态的功能
    @staticmethod 
    def softmax(logits):
        """
        数值稳定 softmax。
        二维数组 logits(样本数,类别数)
        """
        shifted = logits - np.max(logits, axis=1, keepdims=True) #对每个样本（行）减去该行的最大值
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #得到每一个类别的概率

    def forward(self, X,training=True):
        A = X
        # 隐藏层
        for layer in self.layers:
            if isinstance(layer, DropoutLayer):
                A = layer.forward(A, training)
            else:
                A = layer.forward(A)
        # 输出层线性输出
        self.logits = A
        self.probs = self.softmax(self.logits)
        return self.probs

    def compute_loss(self, Y_one_hot):
        """
        交叉熵损失：-1/m * sum(Y_one_hot * log(probs))
        L2 正则化损失：lambda/(2m) * sum(W^2)
        计算损失，接收 one-hot 编码的真实标签
        """
        m = Y_one_hot.shape[0] #样本数
        eps = 1e-12 #防止 log(0) 的数值不稳定问题
        ce_loss = -np.sum(Y_one_hot * np.log(self.probs + eps)) / m
        l2_loss = 0.0
        for layer in self.layers:
            #只对 FullyConnectedLayer 计算 L2 损失，DropoutLayer 没有权重参数
            if isinstance(layer, FullyConnectedLayer):
                l2_loss += np.sum(layer.W ** 2)
        l2_loss *= self.l2_lambda / (2.0 * m)

        return ce_loss + l2_loss

    def backward(self, Y_one_hot):
        """
        反向传播：
        对 softmax + cross-entropy，有 dZ = probs - Y。
        """
        m = Y_one_hot.shape[0]

        # 输出层梯度
        dZ = self.probs - Y_one_hot

        dA_prev = self.layers[-1].backward_from_dZ(dZ, l2_lambda=self.l2_lambda)
        # 隐藏层逆序传播（不包括最后一层）
        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, DropoutLayer):
                dA_prev = layer.backward(dA_prev)          # Dropout 层不需要 l2_lambda
            else:
                dA_prev = layer.backward(dA_prev, l2_lambda=self.l2_lambda)


    def update(self, learning_rate, optimizer="sgd", t=1, beta=0.9, beta1=0.9, beta2=0.999):
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                layer.update_parameters(
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    beta=beta,    # Momentum 的衰减率
                    beta1=beta1,  # Adam 的一阶矩衰减率
                    beta2=beta2,  # Adam 的二阶矩衰减率
                    t=t,           # Adam 的时间步（用于偏差修正）
                )

    def predict_proba(self, X):
        return self.forward(X,training=False)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1) #返回每行（每个样本）中概率最大的索引，即预测的类别标签
