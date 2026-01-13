import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import tensorflow as tf
import plotly.express as px
from tensorflow.keras.layers import Layer, Dense
import tensorflow.keras.backend as K

st.set_page_config(page_title="PrognosAI - RUL Prediction", layout="wide")

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        e = K.tanh(K.dot(inputs, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha_expanded = K.expand_dims(alpha, axis=-1)
        context = inputs * alpha_expanded
        output = K.sum(context, axis=1)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_test_data(seq_file, meta_file):
    sequences = np.load(seq_file)
    metadata = pd.read_csv(meta_file)
    return sequences, metadata

def plot_rul_trends(df_rul):
    st.subheader("📈 Select up to 5 Engines to Display RUL Trends")
    engines = sorted(df_rul['engine_id'].unique())
    selected_engines = st.multiselect("Choose engines:", engines, default=engines[:5], max_selections=5)

    if selected_engines:
        filtered_df = df_rul[df_rul['engine_id'].isin(selected_engines)]
    else:
        filtered_df = df_rul

    fig = px.line(
        filtered_df,
        x='cycle',
        y='RUL',
        color='engine_id',
        labels={'cycle': 'Cycle', 'RUL': 'Predicted RUL (cycles)', 'engine_id': 'Engine ID'},
        hover_name='engine_id',
        title="RUL Degradation Trends"
    )
    fig.update_layout(legend_title_text='Engine ID', height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_alert_zone_counts(df_rul):
    latest = df_rul.groupby('engine_id').tail(1).reset_index(drop=True)
    bins = [-1, 10, 30, float('inf')]
    labels = ['Critical', 'Warning', 'Safe']
    latest['Alert'] = pd.cut(latest['RUL'], bins=bins, labels=labels)
    counts = latest['Alert'].value_counts().reindex(labels).fillna(0).reset_index()
    counts.columns = ['Alert Zone', 'Number of Engines']

    fig = px.bar(
        counts,
        x='Alert Zone',
        y='Number of Engines',
        color='Alert Zone',
        title='🚨 Engine Alert Zone Distribution',
        text='Number of Engines',
        color_discrete_map={'Critical': '#dc3545', 'Warning': '#ffc107', 'Safe': '#28a745'}
    )
    fig.update_traces(textposition='outside', textfont=dict(size=16, color='white'))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def render_dashboard(df_rul):
    st.subheader("📊 Latest RUL Predictions & Maintenance Status")
    latest = df_rul.groupby('engine_id').tail(1).reset_index(drop=True)
    alerts = ['Critical', 'Warning', 'Safe']
    latest['Alert'] = pd.Categorical(pd.cut(latest['RUL'], bins=[-1, 10, 30, float('inf')], labels=alerts))

    col1, col2, col3 = st.columns(3)
    with col1:
        critical_count = (latest['Alert'] == 'Critical').sum()
        st.metric("🔴 Critical", f"{critical_count}")
    with col2:
        warning_count = (latest['Alert'] == 'Warning').sum()
        st.metric("🟡 Warning", f"{warning_count}")
    with col3:
        safe_count = (latest['Alert'] == 'Safe').sum()
        st.metric("🟢 Safe", f"{safe_count}")

    st.markdown("#### 🚨 Maintenance Priority Alerts")
    alert_counts = latest['Alert'].value_counts().reindex(alerts, fill_value=0)
    for level in alerts:
        count = alert_counts[level]
        if count > 0:
            if level == "Critical":
                st.error(f"🚨 URGENT: {count} engine(s) CRITICAL (RUL ≤ 10)")
            elif level == "Warning":
                st.warning(f"⚠️ HIGH PRIORITY: {count} engine(s) WARNING (10-30)")
            else:
                st.success(f"✅ {count} engine(s) SAFE")

    st.markdown("#### 📋 Latest Predictions")
    st.dataframe(latest[['engine_id', 'cycle', 'RUL', 'Alert']].round(1), use_container_width=True)

def main():
    st.title("🛠️ PrognosAI - RUL Prediction Dashboard")
    
    model_path = os.path.join('model', 'bidirectional_attention_model.keras')
    scaler_path = os.path.join('data','processed', 'train', 'scaler.pkl')
    feature_cols_path = os.path.join('data','processed', 'train', 'feature_columns.txt')

    missing_files = [p for p in [model_path, scaler_path, feature_cols_path] if not os.path.exists(p)]
    if missing_files:
        st.error("❌ Missing files:")
        for f in missing_files:
            st.error(f"  • {f}")
        st.info("**Fix:** Run `python data_preprocessing.py` then train model")
        return

    try:
        model, scaler = load_model_and_scaler(model_path, scaler_path)
        st.success("✅ BiLSTM-Attention model loaded!")
        st.info(f"Model input: (None, 30, {model.input_shape[-1]})")
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        st.info("""
        **Quick Fix - Use this exact AttentionLayer in training:**
        ```
        class AttentionLayer(Layer):
            def build(self, input_shape):
                self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
                self.b = self.add_weight(name='attention_bias', shape=(input_shape, 1), initializer='zeros', trainable=True)[1]
                super(AttentionLayer, self).build(input_shape)
        ```
        Retrain with this exact class.
        """)
        return

    col1, col2 = st.columns(2)
    with col1:
        uploaded_seq_file = st.file_uploader("📁 Test Sequences (.npy)", type=["npy"])
    with col2:
        uploaded_meta_file = st.file_uploader("📋 Metadata (.csv)", type=["csv"])

    if uploaded_seq_file and uploaded_meta_file:
        try:
            X_test_seq, df_meta = load_test_data(uploaded_seq_file, uploaded_meta_file)
            st.success(f"✅ Loaded {X_test_seq.shape[0]} sequences: {X_test_seq.shape}")

            if X_test_seq.shape[1:] != (30, model.input_shape[-1]):
                st.error(f"❌ Shape mismatch!")
                st.info(f"Expected: (batch, 30, {model.input_shape[-1]})")
                st.info(f"Got: {X_test_seq.shape}")
                return

            with st.spinner("🔮 Predicting with BiLSTM-Attention..."):
                preds = model.predict(X_test_seq, verbose=0).flatten()
            
            df_meta = df_meta.copy()
            df_meta['RUL'] = preds
            
            st.success(f"✅ Predictions done! Min: {preds.min():.1f}, Max: {preds.max():.1f}, Mean: {preds.mean():.1f}")

            st.markdown("---")
            plot_rul_trends(df_meta)
            st.markdown("---")
            render_dashboard(df_meta)
            st.markdown("---")
            plot_alert_zone_counts(df_meta)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Check file formats match preprocessing output")

if __name__ == "__main__":
    main()
