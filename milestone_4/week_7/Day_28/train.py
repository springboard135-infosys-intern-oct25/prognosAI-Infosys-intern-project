import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================================
# ATTENTIONLAYER
# ============================================================================
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.dense = Dense(1, activation='tanh')
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        attention_scores = self.dense(inputs)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        context_vector = inputs * attention_weights
        output = tf.reduce_sum(context_vector, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(sequence_path, metadata_path):
    """Load data exactly as in your BiDirectional_v2-1.ipynb"""
    X = np.load(sequence_path)
    metadata = pd.read_csv(metadata_path)
    y = metadata['RUL_actual'].values  # Use 'RUL_actual' from metadata
    
    print(f"✓ Loaded data:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Metadata shape: {metadata.shape}")
    
    return X, y, metadata

def create_train_val_split(X, y, metadata, test_size=0.2, random_state=42):
    """Create 80/20 train/val split matching your notebook"""
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X, y, metadata, test_size=test_size, random_state=random_state
    )
    
    print(f"✓ Train/Val split:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    
    return X_train, X_val, y_train, y_val, meta_train, meta_val

# ============================================================================
# MODEL BUILDING
# ============================================================================
def build_bidirectional_attention_model(input_shape=(30, 66), lstm_units=64, dropout_rate=0.3):
    """
    Exact architecture from BiDirectional_v2-1.ipynb:
    Bidirectional(LSTM) → AttentionLayer → Dropout → Dense(1)
    """
    model = Sequential([
        Bidirectional(
            LSTM(lstm_units, activation='tanh', return_sequences=True),
            input_shape=input_shape
        ),
        AttentionLayer(),
        Dropout(dropout_rate),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    
    return model

# ============================================================================
# TRAINING WITH CALLBACKS
# ============================================================================
def train_model(X_train, X_val, y_train, y_val, model_path='models/bidirectional_attention_model.keras'):
    """Training setup matching your notebook exactly"""
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Build model
    model = build_bidirectional_attention_model()
    model.summary()
    
    # Your exact callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            verbose=1, 
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-5, 
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_loss', 
            save_best_only=True, 
            verbose=1
        )
    ]
    
    print("\n🚀 Starting training...")
    print(f"Model will save to: {model_path}")
    
    # Train with tf.data.Dataset (like your notebook)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(1)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64).prefetch(1)
    
    history = model.fit(
        train_dataset,
        epochs=30,  # Like your notebook
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ============================================================================
# SCALER GENERATION FOR DASHBOARD
# ============================================================================
def generate_dashboard_scaler(X_train, output_dir='processed_data/train'):
    """Generate scaler.pkl and feature_columns.txt for Streamlit dashboard"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Fit scaler on TRAINING data only
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)  # (N, 30*66)
    scaler.fit(X_train_flat)
    
    # Save scaler
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    # Generate feature columns list (30 timesteps × 66 features)
    feature_columns = [f't{t}_f{f}' for t in range(30) for f in range(66)]
    with open(f'{output_dir}/feature_columns.txt', 'w') as f:
        f.write('\n'.join(feature_columns))
    
    print(f"\n✅ Dashboard files generated:")
    print(f"   {output_dir}/scaler.pkl")
    print(f"   {output_dir}/feature_columns.txt")
    print(f"   Scaler stats: {scaler.n_features_in_} features")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("🛠️ PrognosAI - BiLSTM Attention Model Training")
    print("=" * 60)
    
    # File paths (matches your preprocessing output)
    sequence_path = 'data/processed/rolling_window_sequences.npy'
    metadata_path = 'data/processed/sequence_metadata_with_RUL.csv'
    
    # Check files exist
    if not os.path.exists(sequence_path):
        print(f"❌ File not found: {sequence_path}")
        print("💡 Run preprocessing pipeline first!")
        return
    
    if not os.path.exists(metadata_path):
        print(f"❌ File not found: {metadata_path}")
        print("💡 Run preprocessing pipeline first!")
        return
    
    # 1. Load data (EXACTLY like your notebook)
    print("\n📂 Loading data...")
    X, y, metadata = load_data(sequence_path, metadata_path)
    
    # 2. Create train/val split (EXACTLY like your notebook)
    X_train, X_val, y_train, y_val, meta_train, meta_val = create_train_val_split(
        X, y, metadata
    )
    
    # 3. Generate dashboard scaler files
    generate_dashboard_scaler(X_train)
    
    # 4. Train model (EXACT architecture + callbacks)
    print("\n🎯 Training BiLSTM-Attention model...")
    model, history = train_model(X_train, X_val, y_train, y_val)
    
    # 5. Save additional files for reproducibility
    np.save('data/processed/train_rolling_window_sequences.npy', X_train)
    np.save('data/processed/val_rolling_window_sequences.npy', X_val)
    meta_train.to_csv('data/processed/train_metadata.csv', index=False)
    meta_val.to_csv('data/processed/val_metadata.csv', index=False)
    
    print("\n🎉 TRAINING COMPLETE!")
    print("\n📁 ALL FILES READY FOR DASHBOARD:")
    print("├── models/bidirectional_attention_model.keras  ✅")
    print("├── processed_data/train/scaler.pkl             ✅")
    print("├── processed_data/train/feature_columns.txt    ✅")
    print("├── data/processed/rolling_window_sequences.npy ✅")
    print("└── data/processed/sequence_metadata_with_RUL.csv ✅")
    
    print("\n🚀 Run: `streamlit run app.py`")

if __name__ == '__main__':
    main()
