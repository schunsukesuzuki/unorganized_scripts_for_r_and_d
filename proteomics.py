"""
プロテオミクス時計 (Proteomics Clock) - JAX実装

タンパク質発現データから生物学的年齢を推定するモデル
ElasticNetベースの線形モデルとニューラルネットワークの両方を実装
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from jax.example_libraries import optimizers
import optax
from typing import Tuple, Dict, Callable
from functools import partial


class ProteomicsClock:
    """
    プロテオミクス時計: タンパク質発現から年齢を予測
    
    ElasticNet回帰ベースの実装
    """
    
    def __init__(self, n_proteins: int, alpha: float = 1.0, l1_ratio: float = 0.5):
        """
        Parameters:
        -----------
        n_proteins: タンパク質の数
        alpha: 正則化強度
        l1_ratio: L1正則化の割合 (0: Ridge, 1: Lasso)
        """
        self.n_proteins = n_proteins
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.params = None
        
    def initialize_params(self, key: jax.random.PRNGKey) -> Dict:
        """パラメータの初期化"""
        weights = random.normal(key, (self.n_proteins,)) * 0.01
        bias = jnp.array(0.0)
        return {'weights': weights, 'bias': bias}
    
    @staticmethod
    @jit
    def predict(params: Dict, X: jnp.ndarray) -> jnp.ndarray:
        """年齢予測"""
        return jnp.dot(X, params['weights']) + params['bias']
    
    def loss_fn(self, params: Dict, X: jnp.ndarray, y: jnp.ndarray) -> float:
        """
        損失関数: MSE + ElasticNet正則化
        """
        predictions = self.predict(params, X)
        mse = jnp.mean((predictions - y) ** 2)
        
        # ElasticNet正則化
        l1_penalty = self.alpha * self.l1_ratio * jnp.sum(jnp.abs(params['weights']))
        l2_penalty = self.alpha * (1 - self.l1_ratio) * jnp.sum(params['weights'] ** 2) / 2
        
        return mse + l1_penalty + l2_penalty
    
    def train(self, 
              X: jnp.ndarray, 
              y: jnp.ndarray, 
              n_epochs: int = 1000,
              learning_rate: float = 0.01,
              key: jax.random.PRNGKey = None) -> Dict:
        """
        モデルの訓練
        
        Parameters:
        -----------
        X: タンパク質発現データ (n_samples, n_proteins)
        y: 年齢データ (n_samples,)
        n_epochs: エポック数
        learning_rate: 学習率
        """
        if key is None:
            key = random.PRNGKey(0)
            
        # パラメータ初期化
        params = self.initialize_params(key)
        
        # オプティマイザ
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)
        
        # 勾配関数
        grad_fn = jit(grad(self.loss_fn))
        
        # 訓練ループ
        losses = []
        for epoch in range(n_epochs):
            grads = grad_fn(params, X, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            if epoch % 100 == 0:
                loss = self.loss_fn(params, X, y)
                losses.append(loss)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        self.params = params
        return {'params': params, 'losses': losses}
    
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict:
        """モデル評価"""
        predictions = self.predict(self.params, X)
        mae = jnp.mean(jnp.abs(predictions - y))
        rmse = jnp.sqrt(jnp.mean((predictions - y) ** 2))
        r2 = 1 - jnp.sum((y - predictions) ** 2) / jnp.sum((y - jnp.mean(y)) ** 2)
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'predictions': predictions
        }


class DeepProteomicsClock:
    """
    深層学習ベースのプロテオミクス時計
    
    多層ニューラルネットワークによる年齢予測
    """
    
    def __init__(self, 
                 n_proteins: int, 
                 hidden_dims: Tuple[int, ...] = (256, 128, 64),
                 dropout_rate: float = 0.2):
        """
        Parameters:
        -----------
        n_proteins: タンパク質の数
        hidden_dims: 隠れ層の次元
        dropout_rate: ドロップアウト率
        """
        self.n_proteins = n_proteins
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.params = None
        
    def initialize_params(self, key: jax.random.PRNGKey) -> list:
        """ネットワークパラメータの初期化"""
        params = []
        dims = [self.n_proteins] + list(self.hidden_dims) + [1]
        
        for i in range(len(dims) - 1):
            key, subkey = random.split(key)
            # Heの初期化
            scale = jnp.sqrt(2.0 / dims[i])
            W = random.normal(subkey, (dims[i], dims[i+1])) * scale
            b = jnp.zeros(dims[i+1])
            params.append({'W': W, 'b': b})
            
        return params
    
    @staticmethod
    def relu(x):
        """ReLU活性化関数"""
        return jnp.maximum(0, x)
    
    def forward(self, params: list, X: jnp.ndarray, training: bool = False, 
                key: jax.random.PRNGKey = None) -> jnp.ndarray:
        """順伝播"""
        activations = X
        
        for i, layer_params in enumerate(params[:-1]):
            activations = jnp.dot(activations, layer_params['W']) + layer_params['b']
            activations = self.relu(activations)
            
            # ドロップアウト (訓練時のみ)
            if training and key is not None:
                key, subkey = random.split(key)
                mask = random.bernoulli(subkey, 1 - self.dropout_rate, activations.shape)
                activations = activations * mask / (1 - self.dropout_rate)
        
        # 出力層
        output = jnp.dot(activations, params[-1]['W']) + params[-1]['b']
        return output.squeeze()
    
    def loss_fn(self, params: list, X: jnp.ndarray, y: jnp.ndarray, 
                key: jax.random.PRNGKey = None) -> float:
        """損失関数"""
        predictions = self.forward(params, X, training=(key is not None), key=key)
        return jnp.mean((predictions - y) ** 2)
    
    def train(self,
              X: jnp.ndarray,
              y: jnp.ndarray,
              X_val: jnp.ndarray = None,
              y_val: jnp.ndarray = None,
              n_epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              key: jax.random.PRNGKey = None) -> Dict:
        """
        モデルの訓練
        
        Parameters:
        -----------
        X: 訓練データのタンパク質発現 (n_samples, n_proteins)
        y: 訓練データの年齢 (n_samples,)
        X_val: 検証データのタンパク質発現
        y_val: 検証データの年齢
        n_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        """
        if key is None:
            key = random.PRNGKey(0)
        
        # パラメータ初期化
        key, subkey = random.split(key)
        params = self.initialize_params(subkey)
        
        # オプティマイザ
        scheduler = optax.exponential_decay(learning_rate, 1000, 0.95)
        optimizer = optax.adam(scheduler)
        opt_state = optimizer.init(params)
        
        # 勾配関数
        def update_step(params, opt_state, X_batch, y_batch, key):
            loss, grads = jax.value_and_grad(self.loss_fn)(params, X_batch, y_batch, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
        
        # 訓練ループ
        n_samples = X.shape[0]
        train_losses = []
        val_losses = []
        
        for epoch in range(n_epochs):
            key, subkey = random.split(key)
            # データシャッフル
            perm = random.permutation(subkey, n_samples)
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            epoch_loss = 0.0
            n_batches = 0
            
            # ミニバッチ訓練
            for i in range(0, n_samples, batch_size):
                key, batch_key = random.split(key)
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                params, opt_state, loss = update_step(
                    params, opt_state, X_batch, y_batch, batch_key
                )
                epoch_loss += loss
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            train_losses.append(float(avg_train_loss))
            
            # 検証
            if X_val is not None and y_val is not None:
                val_predictions = self.forward(params, X_val, training=False)
                val_loss = jnp.mean((val_predictions - y_val) ** 2)
                val_losses.append(float(val_loss))
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}")
        
        self.params = params
        return {
            'params': params,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """年齢予測"""
        return self.forward(self.params, X, training=False)
    
    def evaluate(self, X: jnp.ndarray, y: jnp.ndarray) -> Dict:
        """モデル評価"""
        predictions = self.predict(X)
        mae = jnp.mean(jnp.abs(predictions - y))
        rmse = jnp.sqrt(jnp.mean((predictions - y) ** 2))
        r2 = 1 - jnp.sum((y - predictions) ** 2) / jnp.sum((y - jnp.mean(y)) ** 2)
        
        return {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R2': float(r2),
            'predictions': predictions
        }


def generate_synthetic_proteomics_data(
    n_samples: int = 1000,
    n_proteins: int = 100,
    age_range: Tuple[float, float] = (20, 80),
    noise_level: float = 5.0,
    key: jax.random.PRNGKey = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    合成プロテオミクスデータの生成
    
    Parameters:
    -----------
    n_samples: サンプル数
    n_proteins: タンパク質の数
    age_range: 年齢範囲
    noise_level: ノイズレベル
    
    Returns:
    --------
    X: タンパク質発現データ
    y: 年齢データ
    """
    if key is None:
        key = random.PRNGKey(42)
    
    # 年齢生成
    key, subkey = random.split(key)
    ages = random.uniform(subkey, (n_samples,), minval=age_range[0], maxval=age_range[1])
    
    # 年齢関連タンパク質の数
    n_age_related = n_proteins // 3
    
    # タンパク質発現データ
    key, subkey = random.split(key)
    X = random.normal(subkey, (n_samples, n_proteins))
    
    # 年齢依存性の追加
    for i in range(n_age_related):
        key, subkey = random.split(key)
        # 線形、二次、対数など様々な関係性
        if i % 3 == 0:
            # 線形関係
            X = X.at[:, i].add(ages * random.uniform(subkey, minval=-0.1, maxval=0.1))
        elif i % 3 == 1:
            # 二次関係
            X = X.at[:, i].add((ages ** 2) * random.uniform(subkey, minval=-0.001, maxval=0.001))
        else:
            # 対数関係
            X = X.at[:, i].add(jnp.log(ages + 1) * random.uniform(subkey, minval=-0.5, maxval=0.5))
    
    # ノイズ追加
    key, subkey = random.split(key)
    y_noisy = ages + random.normal(subkey, (n_samples,)) * noise_level
    
    return X, y_noisy


if __name__ == "__main__":
    print("=" * 60)
    print("プロテオミクス時計デモ")
    print("=" * 60)
    
    # データ生成
    key = random.PRNGKey(42)
    X, y = generate_synthetic_proteomics_data(n_samples=500, n_proteins=50, key=key)
    
    # データ分割
    n_train = 400
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"\nデータ形状:")
    print(f"訓練データ: X={X_train.shape}, y={y_train.shape}")
    print(f"テストデータ: X={X_test.shape}, y={y_test.shape}")
    
    # ElasticNetモデル
    print("\n" + "=" * 60)
    print("1. ElasticNetベースのプロテオミクス時計")
    print("=" * 60)
    
    elastic_clock = ProteomicsClock(n_proteins=50, alpha=0.1, l1_ratio=0.5)
    key, subkey = random.split(key)
    elastic_clock.train(X_train, y_train, n_epochs=500, learning_rate=0.01, key=subkey)
    
    elastic_results = elastic_clock.evaluate(X_test, y_test)
    print("\nテスト結果:")
    print(f"MAE: {elastic_results['MAE']:.2f} 歳")
    print(f"RMSE: {elastic_results['RMSE']:.2f} 歳")
    print(f"R²: {elastic_results['R2']:.4f}")
    
    # ディープラーニングモデル
    print("\n" + "=" * 60)
    print("2. 深層学習ベースのプロテオミクス時計")
    print("=" * 60)
    
    deep_clock = DeepProteomicsClock(
        n_proteins=50,
        hidden_dims=(128, 64, 32),
        dropout_rate=0.2
    )
    
    key, subkey = random.split(key)
    deep_clock.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        n_epochs=100,
        batch_size=32,
        learning_rate=0.001,
        key=subkey
    )
    
    deep_results = deep_clock.evaluate(X_test, y_test)
    print("\nテスト結果:")
    print(f"MAE: {deep_results['MAE']:.2f} 歳")
    print(f"RMSE: {deep_results['RMSE']:.2f} 歳")
    print(f"R²: {deep_results['R2']:.4f}")
    
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)