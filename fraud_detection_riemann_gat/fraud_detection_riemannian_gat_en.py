"""
Time Series Anomaly Detection Model with JAX - Fraud Detection (GAT + Riemannian Manifold)
Implementation combining Graph Attention Network (GAT) with Riemannian manifold representations

Features:
- Transaction frequency, average amount, distance, and online flag represented on Riemannian manifolds
- Geometric feature learning through geodesic optimization
- Integration with Euclidean space tensors via pullback operations
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


# ================== Riemannian Manifold Classes ==================

class HyperbolicManifold:
    """Hyperbolic space (Poincaré ball model) - suitable for hierarchical structures"""
    def __init__(self, dim: int, curvature: float = 1.0):
        self.dim = dim
        self.c = curvature  # curvature
    
    def proj(self, x, eps=1e-5):
        """Project onto Poincaré ball (avoid boundary)"""
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        max_norm = (1.0 - eps) / jnp.sqrt(self.c)
        cond = norm > max_norm
        projected = x / norm * max_norm
        return jnp.where(cond, projected, x)
    
    def exp_map(self, x, v):
        """Exponential map: from tangent space to manifold"""
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        norm_v = jnp.maximum(norm_v, 1e-10)
        
        sqrt_c = jnp.sqrt(self.c)
        norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        # Exponential map for Poincaré ball
        lambda_x = 2.0 / (1.0 - self.c * norm_x ** 2)
        
        coef = jnp.tanh(sqrt_c * lambda_x * norm_v / 2) / (sqrt_c * norm_v)
        exp_map = self.mobius_add(x, coef * v)
        
        return self.proj(exp_map)
    
    def log_map(self, x, y):
        """Logarithmic map: from manifold to tangent space"""
        sub = self.mobius_add(-x, y)
        norm_sub = jnp.linalg.norm(sub, axis=-1, keepdims=True)
        norm_sub = jnp.maximum(norm_sub, 1e-10)
        
        sqrt_c = jnp.sqrt(self.c)
        norm_x = jnp.linalg.norm(x, axis=-1, keepdims=True)
        
        lambda_x = 2.0 / (1.0 - self.c * norm_x ** 2)
        
        coef = jnp.arctanh(sqrt_c * norm_sub) / (sqrt_c * norm_sub * lambda_x / 2)
        
        return coef * sub
    
    def mobius_add(self, x, y):
        """Möbius addition"""
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        norm_x_sq = jnp.sum(x ** 2, axis=-1, keepdims=True)
        norm_y_sq = jnp.sum(y ** 2, axis=-1, keepdims=True)
        
        numerator = (1.0 + 2.0 * self.c * xy + self.c * norm_y_sq) * x + \
                   (1.0 - self.c * norm_x_sq) * y
        denominator = 1.0 + 2.0 * self.c * xy + self.c ** 2 * norm_x_sq * norm_y_sq
        
        return numerator / (denominator + 1e-10)
    
    def distance(self, x, y):
        """Hyperbolic distance"""
        sqrt_c = jnp.sqrt(self.c)
        mob_add = self.mobius_add(-x, y)
        norm = jnp.linalg.norm(mob_add, axis=-1)
        return 2.0 / sqrt_c * jnp.arctanh(sqrt_c * norm)


class SPDManifold:
    """Symmetric Positive Definite matrix manifold - suitable for covariance and correlation"""
    def __init__(self, dim: int):
        self.dim = dim
    
    def proj(self, X, eps=1e-6):
        """Project onto SPD manifold"""
        # Symmetrize
        X_sym = 0.5 * (X + X.T)
        # Eigenvalue decomposition
        eigvals, eigvecs = jnp.linalg.eigh(X_sym)
        # Ensure positive definiteness
        eigvals = jnp.maximum(eigvals, eps)
        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
    
    def exp_map(self, X, V):
        """Exponential map (matrix exponential)"""
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
        """Logarithmic map"""
        eigvals_X, eigvecs_X = jnp.linalg.eigh(X)
        sqrt_X = eigvecs_X @ jnp.diag(jnp.sqrt(eigvals_X)) @ eigvecs_X.T
        sqrt_X_inv = eigvecs_X @ jnp.diag(1.0 / jnp.sqrt(eigvals_X)) @ eigvecs_X.T
        
        # log_X(Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}
        W = sqrt_X_inv @ Y @ sqrt_X_inv
        eigvals_W, eigvecs_W = jnp.linalg.eigh(W)
        log_W = eigvecs_W @ jnp.diag(jnp.log(eigvals_W)) @ eigvecs_W.T
        
        return sqrt_X @ log_W @ sqrt_X
    
    def distance(self, X, Y):
        """Riemannian distance"""
        log_map = self.log_map(X, Y)
        return jnp.sqrt(jnp.trace(log_map @ log_map))


class SphereManifold:
    """Sphere manifold - suitable for normalized features"""
    def __init__(self, dim: int):
        self.dim = dim
    
    def proj(self, x, eps=1e-10):
        """Project onto sphere (normalize)"""
        norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x / (norm + eps)
    
    def exp_map(self, x, v):
        """Exponential map"""
        norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)
        norm_v = jnp.maximum(norm_v, 1e-10)
        
        exp_map = jnp.cos(norm_v) * x + jnp.sin(norm_v) * v / norm_v
        return self.proj(exp_map)
    
    def log_map(self, x, y):
        """Logarithmic map"""
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        xy = jnp.clip(xy, -1.0 + 1e-6, 1.0 - 1e-6)
        
        coef = jnp.arccos(xy) / jnp.sqrt(1 - xy ** 2 + 1e-10)
        return coef * (y - xy * x)
    
    def distance(self, x, y):
        """Geodesic distance"""
        xy = jnp.sum(x * y, axis=-1)
        xy = jnp.clip(xy, -1.0 + 1e-6, 1.0 - 1e-6)
        return jnp.arccos(xy)


# ================== Riemannian Feature Encoder ==================

class RiemannianFeatureEncoder(nn.Module):
    """Encoder that represents features on Riemannian manifolds"""
    manifold_dim: int = 8
    
    def setup(self):
        # Initialize each manifold
        self.hyperbolic = HyperbolicManifold(dim=self.manifold_dim, curvature=1.0)
        self.spd = SPDManifold(dim=3)  # 3x3 SPD matrix
        self.sphere = SphereManifold(dim=self.manifold_dim)
    
    @nn.compact
    def __call__(self, features):
        """
        features: [n_nodes, 8] 
        indices:
        0: amount, 1: time_of_day, 2: day_of_week, 3: merchant_category,
        4: transaction_frequency, 5: avg_amount_24h, 6: distance_from_home, 7: online_transaction
        
        Features mapped to Riemannian manifolds:
        - transaction_frequency (4)
        - avg_amount_24h (5)
        - distance_from_home (6)
        - online_transaction (7)
        """
        
        # Euclidean features (used as-is)
        euclidean_features = features[:, :4]  # amount, time, day, merchant
        
        # Features to be mapped to Riemannian manifolds
        manifold_features = features[:, 4:]  # frequency, avg_amount, distance, online
        
        # === Embedding into hyperbolic space (hierarchical relationship of transaction frequency and amount) ===
        # Represent transaction frequency and average amount in hyperbolic space
        freq_amount = jnp.stack([
            manifold_features[:, 0],  # frequency
            manifold_features[:, 1],  # avg_amount
        ], axis=-1)
        
        # Initial embedding into hyperbolic space
        hyperbolic_mlp = nn.Dense(self.manifold_dim, name='hyperbolic_embed')
        hyperbolic_tangent = hyperbolic_mlp(freq_amount)
        
        # Map from origin to hyperbolic space via exponential map
        origin = jnp.zeros((freq_amount.shape[0], self.manifold_dim))
        hyperbolic_point = vmap(self.hyperbolic.exp_map)(origin, hyperbolic_tangent)
        
        # === Embedding into SPD manifold (covariance structure) ===
        # Represent covariance of frequency, amount, distance as 3x3 SPD matrix
        spd_features = jnp.stack([
            manifold_features[:, 0],  # frequency
            manifold_features[:, 1],  # avg_amount
            manifold_features[:, 2],  # distance
        ], axis=-1)
        
        # Build SPD matrix for each sample (simplified version)
        def create_spd_matrix(feat):
            # Make diagonal elements positive to ensure SPD property
            diag_mlp = nn.Dense(3, name='spd_diag')
            diag_elements = jnp.exp(diag_mlp(feat.reshape(1, -1)).squeeze()) + 0.1
            
            # Build as diagonal matrix (for numerical stability)
            S = jnp.diag(diag_elements)
            return S
        
        spd_matrices = vmap(create_spd_matrix)(spd_features)
        
        # === Embedding into sphere (combination of online transaction flag and distance) ===
        sphere_features = jnp.stack([
            manifold_features[:, 2],  # distance
            manifold_features[:, 3],  # online
        ], axis=-1)
        
        sphere_mlp = nn.Dense(self.manifold_dim, name='sphere_embed')
        sphere_unnormalized = sphere_mlp(sphere_features)
        sphere_point = vmap(self.sphere.proj)(sphere_unnormalized)
        
        # === Feature refinement via geodesic optimization ===
        # Optimization in hyperbolic space
        hyperbolic_refined = self.geodesic_refinement_hyperbolic(hyperbolic_point)
        
        # Optimization in SPD manifold
        spd_refined = self.geodesic_refinement_spd(spd_matrices)
        
        # Optimization on sphere
        sphere_refined = self.geodesic_refinement_sphere(sphere_point)
        
        # === Pullback: from Riemannian manifold to Euclidean space ===
        # From hyperbolic space to tangent space (logarithmic map at origin)
        origin = jnp.zeros_like(hyperbolic_refined)
        hyperbolic_pullback = vmap(self.hyperbolic.log_map)(origin, hyperbolic_refined)
        
        # Vectorize SPD matrix in logarithmic domain (simplified version)
        def spd_to_vector(S):
            # Take logarithm of diagonal elements
            diag_elements = jnp.diag(S)
            log_diag = jnp.log(diag_elements + 1e-8)
            return log_diag
        
        spd_pullback = vmap(spd_to_vector)(spd_refined)
        
        # Sphere can be used as-is (already embedded in Euclidean space)
        sphere_pullback = sphere_refined
        
        # === Tensor integration ===
        # Concatenate all representations
        combined_features = jnp.concatenate([
            euclidean_features,      # Original Euclidean features (4D)
            hyperbolic_pullback,     # Pullback from hyperbolic space (manifold_dim D)
            spd_pullback,            # Pullback from SPD manifold (3D)
            sphere_pullback,         # Pullback from sphere (manifold_dim D)
        ], axis=-1)
        
        # Final integrated representation
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
        """Optimization along geodesics in hyperbolic space"""
        # Compute centroid (approximation of Karcher mean)
        centroid = jnp.mean(points, axis=0, keepdims=True)
        centroid = self.hyperbolic.proj(centroid)
        
        # Move along geodesic from each point to centroid
        def refine_point(point):
            tangent = self.hyperbolic.log_map(point, centroid.squeeze())
            # Move slightly toward centroid (alpha=0.1)
            alpha = 0.1
            refined = self.hyperbolic.exp_map(point, alpha * tangent)
            return refined
        
        refined_points = vmap(refine_point)(points)
        return refined_points
    
    def geodesic_refinement_spd(self, matrices):
        """Geodesic optimization in SPD manifold (simplified version)"""
        # Simplified computation of geometric mean (for diagonal matrices)
        # Geometric mean of each diagonal element
        diag_elements = vmap(lambda M: jnp.diag(M))(matrices)
        geo_mean_diag = jnp.exp(jnp.mean(jnp.log(diag_elements + 1e-8), axis=0))
        centroid = jnp.diag(geo_mean_diag)
        
        # Adjust each matrix toward centroid (simplified version)
        def refine_matrix(matrix):
            # Linear interpolation
            alpha = 0.1
            refined = (1 - alpha) * matrix + alpha * centroid
            # Ensure positive definiteness
            eigvals, eigvecs = jnp.linalg.eigh(refined)
            eigvals = jnp.maximum(eigvals, 1e-6)
            return eigvecs @ jnp.diag(eigvals) @ eigvecs.T
        
        refined_matrices = vmap(refine_matrix)(matrices)
        return refined_matrices
    
    def geodesic_refinement_sphere(self, points):
        """Geodesic optimization on sphere"""
        # Mean on sphere (Karcher mean)
        centroid = jnp.mean(points, axis=0, keepdims=True)
        centroid = self.sphere.proj(centroid)
        
        # Adjust along geodesics
        def refine_point(point):
            tangent = self.sphere.log_map(point, centroid.squeeze())
            alpha = 0.1
            refined = self.sphere.exp_map(point, alpha * tangent)
            return refined
        
        refined_points = vmap(refine_point)(points)
        return refined_points


# ================== Data Generation ==================

def generate_transaction_data(
    n_samples: int = 10000,
    n_fraud: int = 500,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dummy credit card transaction data"""
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
    """Normalize data"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def construct_graph(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Construct k-NN graph"""
    A = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    A = A + A.T
    A = (A > 0).astype(np.float32)
    return A.toarray()


# ================== GAT Model Definition (with Riemannian features) ==================

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
    """Anomaly detection model with Riemannian manifolds + GAT"""
    manifold_dim: int = 8
    gat_hidden_dim: int = 32
    gat_out_dim: int = 16
    mlp_hidden_dim: int = 32
    n_heads: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, adj, training: bool = True):
        # Feature encoding on Riemannian manifolds
        riemannian_encoder = RiemannianFeatureEncoder(manifold_dim=self.manifold_dim)
        x_riemannian, manifold_info = riemannian_encoder(x)
        
        # Learn graph structure with GAT layers
        h = GAT(
            hidden_dim=self.gat_hidden_dim,
            out_dim=self.gat_out_dim,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate
        )(x_riemannian, adj, training)
        
        # MLP layers for anomaly score
        h = nn.Dense(self.mlp_hidden_dim)(h)
        h = nn.relu(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(h)
        
        # Reconstruction (back to original input dimension)
        reconstruction = nn.Dense(x.shape[1])(h)
        
        # Anomaly score
        anomaly_score = nn.Dense(1)(h)
        
        return reconstruction, anomaly_score, h, manifold_info


# ================== Training ==================

def create_train_state(rng, model, learning_rate, input_dim, n_nodes):
    """Initialize training state"""
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
    """One training step"""
    dropout_rng = random.fold_in(rng, state.step)
    
    def loss_fn(params):
        rngs = {'dropout': dropout_rng}
        recon_x, anomaly_scores, _, manifold_info = state.apply_fn(
            params, x, adj, training=True, rngs=rngs
        )
        
        # Reconstruction loss
        recon_loss = jnp.mean((x - recon_x) ** 2)
        
        # Anomaly score loss
        y_expanded = y.reshape(-1, 1).astype(jnp.float32)
        classification_loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(anomaly_scores, y_expanded)
        )
        
        # Riemannian geometric regularization term (simplified version)
        # Minimize variance of distances in hyperbolic space
        hyperbolic_points = manifold_info['hyperbolic']
        
        # Centroid of all points
        centroid = jnp.mean(hyperbolic_points, axis=0, keepdims=True)
        
        # Variance of distances from centroid (normal samples should cluster)
        distances = jnp.linalg.norm(hyperbolic_points - centroid, axis=1)
        
        # Weight by label (normal should be small, fraud can be large)
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
    """Train GAT model"""
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
    
    print("Training started...")
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
    
    print("Training completed!")
    return state, history


# ================== Anomaly Detection ==================

def detect_anomalies(state, X, adj, threshold_percentile=95):
    """Detect anomalies"""
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


# ================== Evaluation ==================

def evaluate_model(y_true, y_pred, scores):
    """Evaluate model"""
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\n=== Evaluation Results ===")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    auc = roc_auc_score(y_true, scores)
    print(f"\nROC-AUC Score: {auc:.4f}")
    
    return auc


def plot_results(history, scores, y_true, manifold_info=None, 
                save_path='/mnt/user-data/outputs/fraud_detection_riemannian_gat_results.png'):
    """Visualize results"""
    if manifold_info is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes = axes.flatten()
    
    # Training loss
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
    
    # Anomaly score distribution
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
    
    # ROC curve
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
    
    # Time series of anomaly scores
    ax = axes[3]
    indices = np.arange(len(scores))
    colors = ['blue' if label == 0 else 'red' for label in y_true]
    ax.scatter(indices, scores, c=colors, alpha=0.5, s=10)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Scores (Blue: Normal, Red: Fraud)')
    ax.grid(True, alpha=0.3)
    
    # Manifold visualization
    if manifold_info is not None:
        # 2D projection of hyperbolic space
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
        
        # 2D projection of sphere
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
    print(f"\nResults saved: {save_path}")
    plt.close()


# ================== Main Execution ==================

def main():
    print("=" * 70)
    print("Time Series Anomaly Detection - Fraud Detection (Riemannian Manifold + GAT)")
    print("=" * 70)
    
    print("\n1. Generating data...")
    X, y = generate_transaction_data(n_samples=2000, n_fraud=200, seed=42)
    print(f"Data size: {X.shape}")
    print(f"Normal transactions: {np.sum(y == 0)}, Fraud transactions: {np.sum(y == 1)}")
    
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"\nTraining data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    print("\n2. Normalizing data...")
    X_train_norm, mean, std = normalize_data(X_train)
    X_test_norm = (X_test - mean) / std
    
    print("\n3. Constructing graphs...")
    print("   Training graph...")
    adj_train = construct_graph(X_train_norm, k=10)
    print(f"   Training graph edges: {np.sum(adj_train) / 2:.0f}")
    print("   Test graph...")
    adj_test = construct_graph(X_test_norm, k=10)
    print(f"   Test graph edges: {np.sum(adj_test) / 2:.0f}")
    
    print("\n4. Training Riemannian manifold + GAT model...")
    print("   - Hyperbolic space: hierarchical structure of transaction frequency and average amount")
    print("   - SPD manifold: covariance structure of frequency, amount, and distance")
    print("   - Sphere: relationship between online transactions and distance")
    
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
    
    print("\n5. Detecting anomalies...")
    predictions, scores, threshold, manifold_info = detect_anomalies(
        state, 
        X_test_norm,
        adj_test,
        threshold_percentile=95
    )
    print(f"Anomaly score threshold: {threshold:.4f}")
    
    print("\n6. Evaluating model...")
    auc = evaluate_model(y_test, np.array(predictions), np.array(scores))
    
    print("\n7. Visualizing results (including manifold representations)...")
    plot_results(history, np.array(scores), y_test, manifold_info)
    
    print("\n" + "=" * 70)
    print("Processing completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
