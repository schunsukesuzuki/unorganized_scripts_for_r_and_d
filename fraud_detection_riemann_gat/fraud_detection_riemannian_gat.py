"""
JAX Time Series Anomaly Detection Model - Fraud Detection (GAT + Riemannian Manifold Version)
Implementation combining a Graph Attention Network (GAT) and a Riemannian Manifold representation

Features:
- Transaction frequency, average value, distance, and online flag are represented on a Riemannian manifold
- Geometric feature learning using geodesic optimization
- Integration with Euclidean tensors via pullback
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from functools import partial
from sklearn.neighbors import kneighbors_graph


# ================== リーマン多様体関連 ==================

class HyperbolicManifold:
    """双曲空間（ポアンカレボールモデル）- 階層的構造に適している"""
    def __init__(self, dim: int, curvature: float = 1.0):
        self.dim = dim
        self.c = curvature  # 曲率
    
    def proj(self, x, eps=1e-5):
        """ポアンカレボールへの射影（境界を避ける）"""
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        max_norm = (1.0 - eps) / jnp.sqrt(self.c)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return jnp.where(cond, projected, x)
    
    def exp_map(self, x, v):
        """指数写像: 接空間からマニフォールドへ"""
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        norm_v = jnp.maximum(norm_v, 1e-10)
        
        sqrt_c = jnp.sqrt(self.c)
        norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        # ポアンカレボールの指数写像
        lambda_x = 2.0 / (1.0 - self.c * norm_x ** 2)
        
        coef = jnp.tanh(sqrt_c * lambda_x * norm_v / 2) / (sqrt_c * norm_v)
        exp_map = self.mobius_add(x, coef * v)
        
        return self.proj(exp_map)
    
    def log_map(self, x, y):
        """対数写像: マニフォールドから接空間へ"""
        sub = self.mobius_add(-x, y)
        norm_sub = jnp.linalg.norm(sub, axis=-1, keepdims=True)
        norm_sub = jnp.maximum(norm_sub, 1e-10)
        
        sqrt_c = jnp.sqrt(self.c)
        norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        lambda_x = 2.0 / (1.0 - self.c * norm_x ** 2)
        
        coef = jnp.arctanh(sqrt_c * norm_sub) / (sqrt_c * norm_sub * lambda_x / 2)
        
        return coef * sub
    
    def mobius_add(self, x, y):
        """メビウス加法"""
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        norm_x_sq = jnp.sum(x ** 2, axis=-1, keepdims=True)
        norm_y_sq = jnp.sum(y ** 2, axis=-1, keepdims=True)
        
        numerator = (1.0 + 2.0 * self.c * xy + self.c * norm_y_sq) * x + \
                   (1.0 - self.c * norm_x_sq) * y
        denominator = 1.0 + 2.0 * self.c * xy + self.c ** 2 * norm_x_sq * norm_y_sq
        
        return numerator / (denominator + 1e-10)
    
    def distance(self, x, y):
        """双曲距離"""
        sqrt_c = jnp.sqrt(self.c)
        mob_add = self.mobius_add(-x, y)
        norm = jnp.linalg.norm(mob_add, axis=-1)
        return 2.0 / sqrt_c * jnp.arctanh(sqrt_c * norm)


class SPDManifold:
    """対称正定値行列多様体 - 共分散や相関に適している"""
    def __init__(self, dim: int):
        self.dim = dim
    
    def proj(self, X, eps=1e-6):
        """SPD多様体への射影"""
        # 対称化
        X_sym = 0.5 * (X + X.T)
        # 固有値分解
        eigvals, eigvecs = jnp.linalg.eigh(X_sym)
        # 正定値性を保証
        eigvals = jnp.maximum(eigvals, eps)
        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
    
    def exp_map(self, X, V):
        """指数写像（行列指数関数）"""
        # X^{1/2}
        eigvals, eigvecs = jnp.linalg.eigh(X)
        sqrt_X = eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ eigvecs.T
        sqrt_X_inv = eigvecs @ jnp.diag(1.0 / jnp.sqrt(eigvals)) @ eigvecs.T
        
        # exp_X(V) = X^{1/2} exp(X^{-1/2} V X^{-1/2}) X^{1/2}
        W = sqrt_X_inv @ V @ sqrt_X_inv
        eigvals_W, eigvecs_W = jnp.linalg.eigh(W)
        exp_W = eigvecs_W @ jnp.diag(jnp.exp(eigvals_W)) @ eigvecs_W.T
        
        return sqrt_X @ exp_W @ sqrt_X
    
    def log_map(self, X, Y):
        """対数写像"""
        eigvals_X, eigvecs_X = jnp.linalg.eigh(X)
        sqrt_X = eigvecs_X @ jnp.diag(jnp.sqrt(eigvals_X)) @ eigvecs_X.T
        sqrt_X_inv = eigvecs_X @ jnp.diag(1.0 / jnp.sqrt(eigvals_X)) @ eigvecs_X.T
        
        # log_X(Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}
        W = sqrt_X_inv @ Y @ sqrt_X_inv
        eigvals_W, eigvecs_W = jnp.linalg.eigh(W)
        log_W = eigvecs_W @ jnp.diag(jnp.log(eigvals_W)) @ eigvecs_W.T
        
        return sqrt_X @ log_W @ sqrt_X
    
    def distance(self, X, Y):
        """リーマン距離"""
        log_map = self.log_map(X, Y)
        return jnp.sqrt(jnp.trace(log_map @ log_map))


class SphereManifold:
    """球面多様体 - 正規化された特徴に適している"""
    def __init__(self, dim: int):
        self.dim = dim
    
    def proj(self, x, eps=1e-10):
        """球面への射影（正規化）"""
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + eps)
    
    def exp_map(self, x, v):
        """指数写像"""
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        norm_v = jnp.maximum(norm_v, 1e-10)
        
        exp_map = jnp.cos(norm_v) * x + jnp.sin(norm_v) * v / norm_v
        return self.proj(exp_map)
    
    def log_map(self, x, y):
        """対数写像"""
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        xy = jnp.clip(xy, -1.0 + 1e-6, 1.0 - 1e-6)
        
        coef = jnp.arccos(xy) / jnp.sqrt(1 - xy ** 2 + 1e-10)
        return coef * (y - xy * x)
    
    def distance(self, x, y):
        """測地距離"""
        xy = jnp.sum(x * y, axis=-1)
        xy = jnp.clip(xy, -1.0 + 1e-6, 1.0 - 1e-6)
        return jnp.arccos(xy)


# ================== リーマン多様体上の特徴エンコーダー ==================

class RiemannianFeatureEncoder(nn.Module):
    """リーマン多様体上で特徴を表現するエンコーダー"""
    manifold_dim: int = 8
    
    def setup(self):
        # 各多様体の初期化
        self.hyperbolic = HyperbolicManifold(dim=self.manifold_dim, curvature=1.0)
        self.spd = SPDManifold(dim=3)  # 3x3 SPD行列
        self.sphere = SphereManifold(dim=self.manifold_dim)
    
    @nn.compact
    def __call__(self, features):
        """
        features: [n_nodes, 8] 
        indices:
        0: amount, 1: time_of_day, 2: day_of_week, 3: merchant_category,
        4: transaction_frequency, 5: avg_amount_24h, 6: distance_from_home, 7: online_transaction
        
        リーマン多様体に写像する特徴:
        - transaction_frequency (4)
        - avg_amount_24h (5)
        - distance_from_home (6)
        - online_transaction (7)
        """
        
        # ユークリッド空間の特徴（そのまま使用）
        euclidean_features = features[:, :4]  # amount, time, day, merchant
        
        # リーマン多様体に写像する特徴
        manifold_features = features[:, 4:]  # frequency, avg_amount, distance, online
        
        # === 双曲空間への埋め込み（階層的な取引頻度と金額の関係） ===
        # 取引頻度と平均金額を双曲空間で表現
        freq_amount = jnp.stack([
            manifold_features[:, 0],  # frequency
            manifold_features[:, 1],  # avg_amount
        ], axis=-1)
        
        # 双曲空間への初期埋め込み
        hyperbolic_mlp = nn.Dense(self.manifold_dim, name='hyperbolic_embed')
        hyperbolic_tangent = hyperbolic_mlp(freq_amount)
        
        # 原点から指数写像で双曲空間へ
        origin = jnp.zeros((freq_amount.shape[0], self.manifold_dim))
        hyperbolic_point = vmap(self.hyperbolic.exp_map)(origin, hyperbolic_tangent)
        
        # === SPD多様体への埋め込み（共分散構造） ===
        # 頻度、金額、距離の共分散を3x3 SPD行列で表現
        spd_features = jnp.stack([
            manifold_features[:, 0],  # frequency
            manifold_features[:, 1],  # avg_amount
            manifold_features[:, 2],  # distance
        ], axis=-1)
        
        # 各サンプルごとにSPD行列を構築（簡略化版）
        def create_spd_matrix(feat):
            # 対角要素を正にし、SPD性を保証
            diag_mlp = nn.Dense(3, name='spd_diag')
            diag_elements = jnp.exp(diag_mlp(feat.reshape(1, -1)).squeeze()) + 0.1
            
            # 対角行列として構築（数値安定性のため）
            S = jnp.diag(diag_elements)
            return S
        
        spd_matrices = vmap(create_spd_matrix)(spd_features)
        
        # === 球面への埋め込み（オンライン取引フラグと距離の組み合わせ） ===
        sphere_features = jnp.stack([
            manifold_features[:, 2],  # distance
            manifold_features[:, 3],  # online
        ], axis=-1)
        
        sphere_mlp = nn.Dense(self.manifold_dim, name='sphere_embed')
        sphere_unnormalized = sphere_mlp(sphere_features)
        sphere_point = vmap(self.sphere.proj)(sphere_unnormalized)
        
        # === 測地線最適化による特徴精錬 ===
        # 双曲空間内での最適化
        hyperbolic_refined = self.geodesic_refinement_hyperbolic(hyperbolic_point)
        
        # SPD多様体での最適化
        spd_refined = self.geodesic_refinement_spd(spd_matrices)
        
        # 球面での最適化
        sphere_refined = self.geodesic_refinement_sphere(sphere_point)
        
        # === プルバック: リーマン多様体からユークリッド空間へ ===
        # 双曲空間から接空間へ（原点での対数写像）
        origin = jnp.zeros_like(hyperbolic_refined)
        hyperbolic_pullback = vmap(self.hyperbolic.log_map)(origin, hyperbolic_refined)
        
        # SPD行列を対数領域でベクトル化（簡略版）
        def spd_to_vector(S):
            # 対角要素の対数を取る
            diag_elements = jnp.diag(S)
            log_diag = jnp.log(diag_elements + 1e-8)
            return log_diag
        
        spd_pullback = vmap(spd_to_vector)(spd_refined)
        
        # 球面はそのまま使用可能（すでにユークリッド空間に埋め込まれている）
        sphere_pullback = sphere_refined
        
        # === テンソル統合 ===
        # すべての表現を結合
        combined_features = jnp.concatenate([
            euclidean_features,      # 元のユークリッド特徴 (4次元)
            hyperbolic_pullback,     # 双曲空間からのプルバック (manifold_dim次元)
            spd_pullback,            # SPD多様体からのプルバック (3次元)
            sphere_pullback,         # 球面からのプルバック (manifold_dim次元)
        ], axis=-1)
        
        # 最終的な統合表現
        integration_mlp = nn.Dense(32, name='integration')
        integrated = integration_mlp(combined_features)
        integrated = nn.relu(integrated)
        
        return integrated, {
            'hyperbolic': hyperbolic_refined,
            'spd': spd_refined,
            'sphere': sphere_refined,
            'pullback': combined_features
        }
    
    def geodesic_refinement_hyperbolic(self, points):
        """双曲空間での測地線に沿った最適化"""
        # 重心を計算（Karcher平均の近似）
        centroid = jnp.mean(points, axis=0, keepdims=True)
        centroid = self.hyperbolic.proj(centroid)
        
        # 各点から重心への測地線上で移動
        def refine_point(point):
            tangent = self.hyperbolic.log_map(point, centroid.squeeze())
            # 重心方向に少し移動（alpha=0.1）
            alpha = 0.1
            refined = self.hyperbolic.exp_map(point, alpha * tangent)
            return refined
        
        refined_points = vmap(refine_point)(points)
        return refined_points
    
    def geodesic_refinement_spd(self, matrices):
        """SPD多様体での測地線最適化（簡略版）"""
        # 幾何平均の簡略計算（対角行列の場合）
        # 各対角要素の幾何平均
        diag_elements = vmap(lambda M: jnp.diag(M))(matrices)
        geo_mean_diag = jnp.exp(jnp.mean(jnp.log(diag_elements + 1e-8), axis=0))
        centroid = jnp.diag(geo_mean_diag)
        
        # 各行列を重心方向に調整（簡略版）
        def refine_matrix(matrix):
            # 線形補間で近づける
            alpha = 0.1
            refined = (1 - alpha) * matrix + alpha * centroid
            # 正定値性を保証
            eigvals, eigvecs = jnp.linalg.eigh(refined)
            eigvals = jnp.maximum(eigvals, 1e-6)
            return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        
        refined_matrices = vmap(refine_matrix)(matrices)
        return refined_matrices
    
    def geodesic_refinement_sphere(self, points):
        """球面での測地線最適化"""
        # 球面上の平均（Karcher平均）
        centroid = jnp.mean(points, axis=0, keepdims=True)
        centroid = self.sphere.proj(centroid)
        
        # 測地線に沿って調整
        def refine_point(point):
            tangent = self.sphere.log_map(point, centroid.squeeze())
            alpha = 0.1
            refined = self.sphere.exp_map(point, alpha * tangent)
            return refined
        
        refined_points = vmap(refine_point)(points)
        return refined_points


# ================== データ生成 ==================

def generate_transaction_data(
    n_samples: int = 10000,
    n_fraud: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """クレジットカード取引のダミーデータを生成"""
    np.random.seed(seed)
    
    normal_data = []
    for i in range(n_samples - n_fraud):
        amount = np.random.lognormal(3.5, 1.0)
        time_of_day = np.random.normal(14, 4) % 24
        day_of_week = np.random.choice(7, p=[0.12, 0.14, 0.14, 0.14, 0.16, 0.18, 0.12])
        merchant_category = np.random.choice(10, p=[0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
        transaction_frequency = np.random.poisson(2)
        avg_amount_24h = amount * np.random.uniform(0.8, 1.2)
        distance_from_home = np.random.exponential(10)
        online_transaction = np.random.choice([0, 1], p=[0.6, 0.4])
        
        normal_data.append([
            amount, time_of_day, day_of_week, merchant_category,
            transaction_frequency, avg_amount_24h, distance_from_home, online_transaction
        ])
    
    fraud_data = []
    for i in range(n_fraud):
        amount = np.random.lognormal(5.5, 1.2)
        time_of_day = np.random.choice([0, 1, 2, 3, 4, 23, 22], p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1])
        day_of_week = np.random.choice(7)
        merchant_category = np.random.choice(10)
        transaction_frequency = np.random.poisson(8)
        avg_amount_24h = amount * np.random.uniform(0.5, 0.8)
        distance_from_home = np.random.exponential(50)
        online_transaction = np.random.choice([0, 1], p=[0.3, 0.7])
        
        fraud_data.append([
            amount, time_of_day, day_of_week, merchant_category,
            transaction_frequency, avg_amount_24h, distance_from_home, online_transaction
        ])
    
    X = np.array(normal_data + fraud_data, dtype=np.float32)
    y = np.array([0] * (n_samples - n_fraud) + [1] * n_fraud, dtype=np.int32)
    
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """データを正規化"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def construct_graph(X: np.ndarray, k: int = 10) -> np.ndarray:
    """k-NNグラフを構築"""
    A = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    A = A + A.T
    A = (A > 0).astype(np.float32)
    return A.toarray()


# ================== GATモデル定義（リーマン特徴統合版） ==================

class GATLayer(nn.Module):
    """Graph Attention Layer"""
    out_features: int
    n_heads: int = 4
    dropout_rate: float = 0.1
    concat: bool = True
    
    @nn.compact
    def __call__(self, x, adj, training: bool = True):
        n_nodes = x.shape[0]
        head_outputs = []
        
        for i in range(self.n_heads):
            W = self.param(f'W_{i}', nn.initializers.glorot_uniform(), 
                          (x.shape[1], self.out_features))
            a = self.param(f'a_{i}', nn.initializers.glorot_uniform(), 
                          (2 * self.out_features, 1))
            
            h = jnp.dot(x, W)
            
            h_i = jnp.repeat(h[:, None, :], n_nodes, axis=1)
            h_j = jnp.repeat(h[None, :, :], n_nodes, axis=0)
            h_concat = jnp.concatenate([h_i, h_j], axis=-1)
            
            e = jnp.dot(h_concat, a).squeeze(-1)
            e = nn.leaky_relu(e, negative_slope=0.2)
            
            mask = (adj == 0).astype(jnp.float32) * -1e9
            e = e + mask
            
            alpha = nn.softmax(e, axis=-1)
            alpha = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(alpha)
            
            h_out = jnp.dot(alpha, h)
            head_outputs.append(h_out)
        
        if self.concat:
            output = jnp.concatenate(head_outputs, axis=-1)
        else:
            output = jnp.mean(jnp.stack(head_outputs), axis=0)
        
        return output


class GAT(nn.Module):
    """Graph Attention Network"""
    hidden_dim: int = 32
    out_dim: int = 16
    n_heads: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, adj, training: bool = True):
        x = GATLayer(
            out_features=self.hidden_dim // self.n_heads,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            concat=True
        )(x, adj, training)
        x = nn.elu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        
        x = GATLayer(
            out_features=self.out_dim,
            n_heads=1,
            dropout_rate=self.dropout_rate,
            concat=False
        )(x, adj, training)
        
        return x


class RiemannianGATAnomalyDetector(nn.Module):
    """リーマン多様体 + GAT による異常検知モデル"""
    manifold_dim: int = 8
    gat_hidden_dim: int = 32
    gat_out_dim: int = 16
    mlp_hidden_dim: int = 32
    n_heads: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, adj, training: bool = True):
        # リーマン多様体上での特徴エンコーディング
        riemannian_encoder = RiemannianFeatureEncoder(manifold_dim=self.manifold_dim)
        x_riemannian, manifold_info = riemannian_encoder(x)
        
        # GAT層でグラフ構造を学習
        h = GAT(
            hidden_dim=self.gat_hidden_dim,
            out_dim=self.gat_out_dim,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
        )(x_riemannian, adj, training)
        
        # MLP層で異常スコアを出力
        h = nn.Dense(self.mlp_hidden_dim)(h)
        h = nn.relu(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
        
        # 再構成（元の入力次元に戻す）
        reconstruction = nn.Dense(x.shape[1])(h)
        
        # 異常スコア
        anomaly_score = nn.Dense(1)(h)
        
        return reconstruction, anomaly_score, h, manifold_info


# ================== 訓練 ==================

def create_train_state(rng, model, learning_rate, input_dim, n_nodes):
    """訓練状態を初期化"""
    dummy_x = jnp.ones([n_nodes, input_dim])
    dummy_adj = jnp.ones([n_nodes, n_nodes])
    params = model.init({'params': rng, 'dropout': rng}, dummy_x, dummy_adj, training=False)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )


@jit
def train_step(state, x, adj, y, rng):
    """1ステップの訓練"""
    dropout_rng = random.fold_in(rng, state.step)
    
    def loss_fn(params):
        rngs = {'dropout': dropout_rng}
        recon_x, anomaly_scores, _, manifold_info = state.apply_fn(
            params, x, adj, training=True, rngs=rngs
        )
        
        # 再構成誤差
        recon_loss = jnp.mean((x - recon_x) ** 2)
        
        # 異常スコアの損失
        y_expanded = y.reshape(-1, 1).astype(jnp.float32)
        classification_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(anomaly_scores, y_expanded)
        )
        
        # リーマン幾何学的な正則化項（簡略化版）
        # 双曲空間での距離の分散を最小化
        hyperbolic_points = manifold_info['hyperbolic']
        
        # 全点の重心
        centroid = jnp.mean(hyperbolic_points, axis=0, keepdims=True)
        
        # 重心からの距離の分散（正常サンプルは密集すべき）
        distances = jnp.linalg.norm(hyperbolic_points - centroid, axis=1)
        
        # ラベルで重み付け（正常は小さく、異常は大きく）
        weights = jnp.where(y == 0, 1.0, 0.1)
        weighted_distances = distances * weights
        geo_reg = jnp.mean(weighted_distances)
        
        total_loss = recon_loss + classification_loss + 0.01 * geo_reg
        
        return total_loss, (recon_loss, classification_loss, geo_reg)
    
    (loss, (recon_loss, class_loss, geo_reg)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, recon_loss, class_loss, geo_reg


def train_gat(
    X_train: np.ndarray,
    y_train: np.ndarray,
    adj_train: np.ndarray,
    manifold_dim: int = 8,
    gat_hidden_dim: int = 32,
    gat_out_dim: int = 16,
    mlp_hidden_dim: int = 32,
    n_heads: int = 4,
    learning_rate: float = 1e-3,
    n_epochs: int = 200,
    dropout_rate: float = 0.1,
    seed: int = 0
):
    """GATを訓練"""
    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)
    
    n_nodes, input_dim = X_train.shape
    model = RiemannianGATAnomalyDetector(
        manifold_dim=manifold_dim,
        gat_hidden_dim=gat_hidden_dim,
        gat_out_dim=gat_out_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        n_heads=n_heads,
        dropout_rate=dropout_rate
    )
    
    state = create_train_state(init_rng, model, learning_rate, input_dim, n_nodes)
    
    X_train_jax = jnp.array(X_train)
    y_train_jax = jnp.array(y_train)
    adj_train_jax = jnp.array(adj_train)
    
    history = {'loss': [], 'recon_loss': [], 'class_loss': [], 'geo_reg': []}
    
    print("訓練開始...")
    for epoch in range(n_epochs):
        rng, step_rng = random.split(rng)
        
        state, loss, recon_loss, class_loss, geo_reg = train_step(
            state, X_train_jax, adj_train_jax, y_train_jax, step_rng
        )
        
        history['loss'].append(float(loss))
        history['recon_loss'].append(float(recon_loss))
        history['class_loss'].append(float(class_loss))
        history['geo_reg'].append(float(geo_reg))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.4f}, "
                  f"Recon: {recon_loss:.4f}, Class: {class_loss:.4f}, Geo: {geo_reg:.4f}")
    
    print("訓練完了!")
    return state, history


# ================== 異常検知 ==================

def detect_anomalies(state, X, adj, threshold_percentile=95):
    """異常を検知"""
    X_jax = jnp.array(X)
    adj_jax = jnp.array(adj)
    
    recon_X, anomaly_scores, _, manifold_info = state.apply_fn(
        state.params, X_jax, adj_jax, training=False
    )
    
    recon_errors = jnp.mean((X_jax - recon_X) ** 2, axis=1)
    anomaly_probs = jax.nn.sigmoid(anomaly_scores.squeeze())
    
    combined_scores = 0.5 * recon_errors + 0.5 * anomaly_probs
    
    threshold = jnp.percentile(combined_scores, threshold_percentile)
    predictions = (combined_scores > threshold).astype(int)
    
    return predictions, combined_scores, threshold, manifold_info


# ================== 評価 ==================

def evaluate_model(y_true, y_pred, scores):
    """モデルを評価"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\n=== 評価結果 ===")
    print("\n混同行列:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\n分類レポート:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    auc = roc_auc_score(y_true, scores)
    print(f"\nROC-AUC Score: {auc:.4f}")
    
    return auc


def plot_results(history, scores, y_true, manifold_info=None, 
                save_path='/mnt/user-data/outputs/fraud_detection_riemannian_gat_results.png'):
    """結果を可視化"""
    if manifold_info is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes = axes.flatten()
    
    # 損失の推移
    ax = axes[0]
    ax.plot(history['loss'], label='Total Loss', linewidth=2)
    ax.plot(history['recon_loss'], label='Reconstruction Loss', alpha=0.7)
    ax.plot(history['class_loss'], label='Classification Loss', alpha=0.7)
    if 'geo_reg' in history:
        ax.plot(history['geo_reg'], label='Geometric Regularization', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss (Riemannian GAT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 異常スコアの分布
    ax = axes[1]
    normal_scores = scores[y_true == 0]
    fraud_scores = scores[y_true == 1]
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', density=True)
    ax.hist(fraud_scores, bins=50, alpha=0.6, label='Fraud', density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title('Anomaly Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC曲線
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    ax = axes[2]
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 異常スコアの時系列
    ax = axes[3]
    indices = np.arange(len(scores))
    colors = ['blue' if label == 0 else 'red' for label in y_true]
    ax.scatter(indices, scores, c=colors, alpha=0.5, s=10)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Scores (Blue: Normal, Red: Fraud)')
    ax.grid(True, alpha=0.3)
    
    # 多様体可視化
    if manifold_info is not None:
        # 双曲空間の2次元投影
        ax = axes[4]
        hyperbolic_2d = np.array(manifold_info['hyperbolic'][:, :2])
        colors = ['blue' if label == 0 else 'red' for label in y_true]
        ax.scatter(hyperbolic_2d[:, 0], hyperbolic_2d[:, 1], c=colors, alpha=0.5, s=20)
        circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title('Hyperbolic Space Projection')
        ax.grid(True, alpha=0.3)
        
        # 球面の2次元投影
        ax = axes[5]
        sphere_2d = np.array(manifold_info['sphere'][:, :2])
        ax.scatter(sphere_2d[:, 0], sphere_2d[:, 1], c=colors, alpha=0.5, s=20)
        circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linestyle='--')
        ax.add_patch(circle)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title('Sphere Manifold Projection')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n結果を保存しました: {save_path}")
    plt.close()


# ================== メイン実行 ==================

def main():
    print("=" * 70)
    print("JAX時系列異常検知モデル - Fraud Detection (Riemannian Manifold + GAT)")
    print("=" * 70)
    
    print("\n1. データ生成中...")
    X, y = generate_transaction_data(n_samples=2000, n_fraud=200, seed=42)
    print(f"データサイズ: {X.shape}")
    print(f"正常取引: {np.sum(y == 0)}, 不正取引: {np.sum(y == 1)}")
    
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\n訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")
    
    print("\n2. データ正規化中...")
    X_train_norm, mean, std = normalize_data(X_train)
    X_test_norm = (X_test - mean) / std
    
    print("\n3. グラフ構築中...")
    print("   訓練グラフ...")
    adj_train = construct_graph(X_train_norm, k=10)
    print(f"   訓練グラフエッジ数: {np.sum(adj_train) / 2:.0f}")
    print("   テストグラフ...")
    adj_test = construct_graph(X_test_norm, k=10)
    print(f"   テストグラフエッジ数: {np.sum(adj_test) / 2:.0f}")
    
    print("\n4. リーマン多様体 + GAT モデル訓練中...")
    print("   - 双曲空間: 取引頻度と平均金額の階層構造")
    print("   - SPD多様体: 頻度・金額・距離の共分散構造")
    print("   - 球面: オンライン取引と距離の関係")
    
    state, history = train_gat(
        X_train_norm,
        y_train,
        adj_train,
        manifold_dim=8,
        gat_hidden_dim=32,
        gat_out_dim=16,
        mlp_hidden_dim=32,
        n_heads=4,
        learning_rate=1e-3,
        n_epochs=200,
        dropout_rate=0.1,
        seed=42
    )
    
    print("\n5. 異常検知中...")
    predictions, scores, threshold, manifold_info = detect_anomalies(
        state, 
        X_test_norm,
        adj_test,
        threshold_percentile=95
    )
    print(f"異常スコア閾値: {threshold:.4f}")
    
    print("\n6. モデル評価中...")
    auc = evaluate_model(y_test, np.array(predictions), np.array(scores))
    
    print("\n7. 結果可視化中（多様体表現を含む）...")
    plot_results(history, np.array(scores), y_test, manifold_info)
    
    print("\n" + "=" * 70)
    print("処理完了!")
    print("=" * 70)


if __name__ == "__main__":
    main()
