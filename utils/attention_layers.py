"""
JetX Predictor - Attention Mechanism Layers

Multi-Head Attention ve Transformer-like attention layers.
Model performansını artırmak için sequence modellerine eklenebilir.
"""

class PositionalEncoding(layers.Layer):
    """
    Positional Encoding for Transformer
    Time series için zamansal bilgi ekler
    """
    def __init__(self, max_seq_len=1000, d_model=256, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Positional encoding matrix oluştur
        position = tf.range(max_seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        
        pe = tf.zeros((max_seq_len, d_model))
        pe_sin = tf.sin(position * div_term)
        pe_cos = tf.cos(position * div_term)
        
        # Sin ve cos değerlerini birleştir
        pe_array = tf.Variable(pe, trainable=False)
        pe_array[:, 0::2].assign(pe_sin)
        pe_array[:, 1::2].assign(pe_cos)
        self.pe = pe_array
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'd_model': self.d_model
        })
        return config


class LightweightTransformerEncoder(layers.Layer):
    """
    Lightweight Transformer Encoder for Time Series
    
    Args:
        d_model: Model dimension (256)
        num_layers: Number of transformer layers (4)
        num_heads: Number of attention heads (8)
        dff: Feedforward dimension (1024)
        dropout: Dropout rate (0.2)
    """
    def __init__(
        self, 
        d_model=256, 
        num_layers=4, 
        num_heads=8, 
        dff=1024, 
        dropout=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout
        
        # Input projection (sequence_len, 1) → (sequence_len, d_model)
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len=1000, d_model=d_model)
        
        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            # Multi-head attention
            mha = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout
            )
            
            # Feedforward network
            ffn = tf.keras.Sequential([
                layers.Dense(dff, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(d_model)
            ])
            
            # Layer normalization
            layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            layernorm2 = layers.LayerNormalization(epsilon=1e-6)
            
            # Dropout
            dropout1 = layers.Dropout(dropout)
            dropout2 = layers.Dropout(dropout)
            
            self.encoder_layers.append({
                'mha': mha,
                'ffn': ffn,
                'layernorm1': layernorm1,
                'layernorm2': layernorm2,
                'dropout1': dropout1,
                'dropout2': dropout2
            })
        
        # Global average pooling
        self.global_pool = layers.GlobalAveragePooling1D()
        
        # Output projection
        self.output_projection = layers.Dense(d_model)
        self.dropout_final = layers.Dropout(dropout)
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: (batch_size, seq_len, 1) - Time series input
            training: Training mode flag
            
        Returns:
            (batch_size, d_model) - Encoded representation
        """
        # Input projection
        x = self.input_projection(inputs)  # (batch, seq_len, d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoder layers
        for layer in self.encoder_layers:
            # Multi-head attention
            attn_output = layer['mha'](
                query=x,
                key=x,
                value=x,
                training=training
            )
            attn_output = layer['dropout1'](attn_output, training=training)
            x = layer['layernorm1'](x + attn_output)  # Residual connection
            
            # Feedforward network
            ffn_output = layer['ffn'](x)
            ffn_output = layer['dropout2'](ffn_output, training=training)
            x = layer['layernorm2'](x + ffn_output)  # Residual connection
        
        # Global pooling
        x = self.global_pool(x)  # (batch, d_model)
        
        # Output projection
        x = self.output_projection(x)
        x = self.dropout_final(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout': self.dropout_rate
        })
        return config
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from typing import Optional


class MultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention Layer
    
    Transformer mimarisinden ilham alınmıştır.
    Farklı representation subspace'lerden bilgi toplar.
    
    Args:
        num_heads: Attention head sayısı
        key_dim: Her head için key/query dimension
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_heads: int = 8,
        key_dim: int = 64,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout
        
        # Multi-head attention layer
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )
        
        # Add & Norm
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Input tensor (batch_size, seq_len, d_model)
            training: Training mode flag
            
        Returns:
            Output tensor (batch_size, seq_len, d_model)
        """
        # Self-attention
        attn_output = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )
        
        # Dropout
        attn_output = self.dropout(attn_output, training=training)
        
        # Add & Norm (residual connection)
        output = self.layernorm(inputs + attn_output)
        
        return output
    
    def get_config(self):
        """Config for serialization"""
        config = super().get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout_rate
        })
        return config


class SelfAttention(layers.Layer):
    """
    Simple Self-Attention Layer
    
    Lightweight attention mechanism.
    
    Args:
        units: Attention units (dimension)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        units: int = 128,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        
        # Query, Key, Value projections
        self.query_dense = layers.Dense(units)
        self.key_dense = layers.Dense(units)
        self.value_dense = layers.Dense(units)
        
        # Output projection
        self.output_dense = layers.Dense(units)
        
        # Dropout
        self.dropout = layers.Dropout(dropout)
        
        # Layer norm
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Input tensor (batch_size, seq_len, d_model)
            training: Training mode flag
            
        Returns:
            Output tensor (batch_size, seq_len, units)
        """
        # Projections
        query = self.query_dense(inputs)  # (batch, seq_len, units)
        key = self.key_dense(inputs)      # (batch, seq_len, units)
        value = self.value_dense(inputs)  # (batch, seq_len, units)
        
        # Attention scores
        # (batch, seq_len, units) @ (batch, units, seq_len) = (batch, seq_len, seq_len)
        scores = tf.matmul(query, key, transpose_b=True)
        
        # Scale
        dk = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_scores = scores / tf.math.sqrt(dk)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_scores, axis=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        # (batch, seq_len, seq_len) @ (batch, seq_len, units) = (batch, seq_len, units)
        context = tf.matmul(attention_weights, value)
        
        # Output projection
        output = self.output_dense(context)
        
        # Dropout
        output = self.dropout(output, training=training)
        
        # Add & Norm
        output = self.layernorm(inputs + output)
        
        return output
    
    def get_config(self):
        """Config for serialization"""
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout_rate
        })
        return config


class TemporalAttention(layers.Layer):
    """
    Temporal Attention for Time Series
    
    Zaman serileri için özelleştirilmiş attention.
    Son değerlere daha fazla odaklanır.
    
    Args:
        units: Attention units
        dropout: Dropout rate
        recency_bias: Son değerlere bias (0-1 arası)
    """
    
    def __init__(
        self,
        units: int = 128,
        dropout: float = 0.1,
        recency_bias: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        self.recency_bias = recency_bias
        
        # Attention mechanism
        self.query_dense = layers.Dense(units)
        self.key_dense = layers.Dense(units)
        self.value_dense = layers.Dense(units)
        
        # Context vector
        self.context_vector = self.add_weight(
            name='context_vector',
            shape=(units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.dropout = layers.Dropout(dropout)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Input tensor (batch_size, seq_len, d_model)
            training: Training mode flag
            
        Returns:
            Output tensor (batch_size, units)
        """
        # Projections
        query = self.query_dense(inputs)  # (batch, seq_len, units)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        
        # Attention scores using context vector
        # (batch, seq_len, units) @ (units, 1) = (batch, seq_len, 1)
        scores = tf.matmul(
            tf.nn.tanh(query),
            self.context_vector
        )
        
        # Recency bias: Son timestep'lere daha fazla ağırlık
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len, dtype=tf.float32)
        positions = positions / tf.cast(seq_len - 1, tf.float32)  # 0 to 1
        recency_weights = 1.0 + self.recency_bias * positions
        recency_weights = tf.reshape(recency_weights, (1, -1, 1))
        
        scores = scores * recency_weights
        
        # Softmax
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Weighted sum
        # (batch, seq_len, units) * (batch, seq_len, 1) = (batch, seq_len, units)
        # sum over seq_len = (batch, units)
        context = tf.reduce_sum(value * attention_weights, axis=1)
        
        return context
    
    def get_config(self):
        """Config for serialization"""
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout_rate,
            'recency_bias': self.recency_bias
        })
        return config


class AdditiveAttention(layers.Layer):
    """
    Additive (Bahdanau) Attention
    
    Score function: v^T * tanh(W1*h + W2*s)
    
    Args:
        units: Attention units
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        units: int = 128,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        
        # Attention weights
        self.W1 = layers.Dense(units, use_bias=False)
        self.W2 = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)
        
        self.dropout = layers.Dropout(dropout)
    
    def call(self, query, values, training=None):
        """
        Forward pass
        
        Args:
            query: Query tensor (batch_size, query_dim)
            values: Values tensor (batch_size, seq_len, value_dim)
            training: Training mode flag
            
        Returns:
            context: Context vector (batch_size, value_dim)
            attention_weights: Attention weights (batch_size, seq_len, 1)
        """
        # Expand query to match values shape
        # (batch, query_dim) -> (batch, 1, units)
        query_with_time_axis = tf.expand_dims(query, 1)
        
        # Score calculation
        # W1(values): (batch, seq_len, units)
        # W2(query): (batch, 1, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(query_with_time_axis)
        ))  # (batch, seq_len, 1)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Context vector
        context = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context, attention_weights
    
    def get_config(self):
        """Config for serialization"""
        config = super().get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout_rate
        })
        return config


# Kullanım örneği
def create_attention_enhanced_model(
    input_shape: tuple,
    attention_type: str = 'multi_head',
    num_heads: int = 4,
    key_dim: int = 64
):
    """
    Attention-enhanced model örneği
    
    Args:
        input_shape: Input shape (seq_len, features)
        attention_type: 'multi_head', 'self', 'temporal'
        num_heads: Multi-head için head sayısı
        key_dim: Key dimension
        
    Returns:
        Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Sequence processing (örnek olarak LSTM)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    
    # Attention layer
    if attention_type == 'multi_head':
        x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x)
    elif attention_type == 'self':
        x = SelfAttention(units=128)(x)
    elif attention_type == 'temporal':
        x = TemporalAttention(units=128)(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Output
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


if __name__ == "__main__":
    print("🎯 Attention Layers Test")
    
    # Test multi-head attention
    print("\n1. Multi-Head Attention")
    mha = MultiHeadAttention(num_heads=4, key_dim=64)
    dummy_input = tf.random.normal((2, 10, 128))  # (batch, seq_len, features)
    output = mha(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test self attention
    print("\n2. Self Attention")
    sa = SelfAttention(units=128)
    output = sa(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test temporal attention
    print("\n3. Temporal Attention")
    ta = TemporalAttention(units=128, recency_bias=0.3)
    output = ta(dummy_input)
    print(f"Output shape: {output.shape}")
    
    print("\n✅ Tüm attention layers başarıyla test edildi!")
