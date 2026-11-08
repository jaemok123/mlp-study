"""순수 NumPy로 구현한 MNIST MLP (784-256-128-10).

PyTorch 없이 순전파, 역전파, Adam 옵티마이저를 직접 구현한다.
구조는 step3_models.py의 MLP와 같다: Linear-ReLU-Linear-ReLU-Linear.

사용:
  python numpy_mlp.py            # 학습 + 평가 + numpy_mlp.npz 저장
  python numpy_mlp.py --check    # 역전파 기울기 검증(수치 미분과 비교)만 실행
"""
import argparse
import gzip
import os
import struct
import time
import urllib.request

import numpy as np

# ----------------------------------------------------------------------
# 1. 데이터 불러오기 (torchvision 없이 MNIST 원본 파일을 직접 읽음)
# ----------------------------------------------------------------------
FILES = {
    "train_x": "train-images-idx3-ubyte",
    "train_y": "train-labels-idx1-ubyte",
    "test_x": "t10k-images-idx3-ubyte",
    "test_y": "t10k-labels-idx1-ubyte",
}
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]


def read_idx(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        ndim = struct.unpack(">I", f.read(4))[0] & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def find_or_download(name):
    # 이전에 torchvision이 받아 둔 파일이 있으면 그대로 사용
    local = os.path.join("data", "MNIST", "raw", name)
    if os.path.exists(local):
        return local
    if os.path.exists(local + ".gz"):
        return local + ".gz"
    os.makedirs("data/numpy_mnist", exist_ok=True)
    dest = os.path.join("data", "numpy_mnist", name + ".gz")
    if not os.path.exists(dest):
        for base in MIRRORS:
            try:
                print("다운로드:", base + name + ".gz")
                urllib.request.urlretrieve(base + name + ".gz", dest)
                break
            except Exception as e:
                print("  실패:", e)
        else:
            raise RuntimeError(f"{name} 다운로드 실패")
    return dest


def load_mnist():
    d = {k: read_idx(find_or_download(v)) for k, v in FILES.items()}
    x_train = d["train_x"].reshape(-1, 784).astype(np.float32) / 255.0
    x_test = d["test_x"].reshape(-1, 784).astype(np.float32) / 255.0
    # PyTorch 코드와 같은 정규화 (평균 0.1307, 표준편차 0.3081)
    x_train = (x_train - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081
    return x_train, d["train_y"].astype(np.int64), x_test, d["test_y"].astype(np.int64)


# ----------------------------------------------------------------------
# 2. 모델
# ----------------------------------------------------------------------
class NumpyMLP:
    def __init__(self, sizes=(784, 256, 128, 10), seed=0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # He 초기화: ReLU 층에서 신호 크기가 층마다 줄거나 커지지 않게 함
            self.W.append((rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(dtype))
            self.b.append(np.zeros(n_out, dtype=dtype))
        # Adam 상태
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        # [W1, b1, W2, b2, W3, b3] 순서
        return [p for pair in zip(self.W, self.b) for p in pair]

    # ---- 순전파 ----
    def forward(self, x):
        self.cache = [x]               # 각 층의 입력을 저장 (역전파에서 사용)
        self.masks = []
        a = x
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b              # 선형 변환
            if i < last:
                mask = z > 0           # ReLU가 살린 위치 기록
                a = z * mask
                self.masks.append(mask)
                self.cache.append(a)
            else:
                a = z                  # 마지막 층은 점수(로짓) 그대로 출력
        return a

    # ---- 손실 (softmax + cross entropy, 수치 안정화 포함) ----
    @staticmethod
    def softmax(logits):
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def loss(self, logits, y):
        p = self.softmax(logits)
        return -np.log(p[np.arange(len(y)), y] + 1e-12).mean()

    # ---- 역전파 ----
    def backward(self, logits, y):
        n = len(y)
        # softmax + cross entropy의 기울기는 (확률 - 정답 원-핫) / 배치 크기
        dz = self.softmax(logits)
        dz[np.arange(n), y] -= 1.0
        dz /= n

        grads = [None] * len(self.params())
        for i in reversed(range(len(self.W))):
            a_in = self.cache[i]
            grads[2 * i] = a_in.T @ dz              # dL/dW
            grads[2 * i + 1] = dz.sum(axis=0)       # dL/db
            if i > 0:
                da = dz @ self.W[i].T               # 앞 층으로 오차 전달
                dz = da * self.masks[i - 1]         # ReLU 미분: 살아 있던 위치만 통과
        return grads

    # ---- Adam 갱신 ----
    def step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k, (p, g) in enumerate(zip(self.params(), grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)   # p는 제자리 수정됨

    def predict(self, x):
        return self.forward(x).argmax(axis=1)

    def save(self, path):
        np.savez(path, **{f"p{i}": p for i, p in enumerate(self.params())})


# ----------------------------------------------------------------------
# 3. 역전파 검증: 수치 미분과 비교
# ----------------------------------------------------------------------
def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    model = NumpyMLP(sizes=(20, 16, 12, 10), seed=seed, dtype=np.float64)
    x = rng.standard_normal((8, 20))
    y = rng.integers(0, 10, size=8)

    logits = model.forward(x)
    grads = model.backward(logits, y)

    worst = 0.0
    h = 1e-6
    for p, g in zip(model.params(), grads):
        flat, gflat = p.reshape(-1), g.reshape(-1)
        for idx in rng.choice(flat.size, size=min(10, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + h
            lp = model.loss(model.forward(x), y)
            flat[idx] = old - h
            lm = model.loss(model.forward(x), y)
            flat[idx] = old
            num = (lp - lm) / (2 * h)
            rel = abs(num - gflat[idx]) / max(1e-8, abs(num) + abs(gflat[idx]))
            worst = max(worst, rel)
    print(f"기울기 검증: 최대 상대 오차 {worst:.2e} ({'통과' if worst < 1e-5 else '실패'})")
    return worst


# ----------------------------------------------------------------------
# 4. 학습
# ----------------------------------------------------------------------
def evaluate(model, x, y, batch=2000):
    correct = 0
    for i in range(0, len(x), batch):
        correct += (model.predict(x[i:i + batch]) == y[i:i + batch]).sum()
    return correct / len(x)


def train(model, x_train, y_train, x_test, y_test, epochs=5, batch=128, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x_train)
    for epoch in range(1, epochs + 1):
        start = time.time()
        order = rng.permutation(n)                     # 매 에포크 순서 섞기
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb, yb = x_train[idx], y_train[idx]
            logits = model.forward(xb)                 # 1) 순전파
            total += model.loss(logits, yb) * len(idx) # 2) 손실
            grads = model.backward(logits, yb)         # 3) 역전파
            model.step(grads, lr=lr)                   # 4) 가중치 갱신
        acc = evaluate(model, x_test, y_test)
        print(f"epoch {epoch}/{epochs} loss {total / n:.4f} | 테스트 정확도 {acc:.2%} | {time.time() - start:.1f}초")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="기울기 검증만 실행")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    gradient_check()
    if not args.check:
        x_train, y_train, x_test, y_test = load_mnist()
        print("학습 데이터:", x_train.shape, "테스트 데이터:", x_test.shape)
        model = NumpyMLP()
        train(model, x_train, y_train, x_test, y_test, epochs=args.epochs)
        model.save("numpy_mlp.npz")
        print("저장: numpy_mlp.npz")
