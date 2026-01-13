import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import os
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, window_size=30, scaler_type='standard'):
        self.window_size = window_size
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_cols = None

    def load_data(self, file_path, dataset_type='train'):
        cols = ['engine_id', 'cycle'] + \
               [f'op_setting_{i}' for i in range(1, 4)] + \
               [f'sensor_{i}' for i in range(1, 22)]
        if file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=cols)
        else:
            df = pd.read_csv(file_path)
        print(f"Loaded {dataset_type} data: {df.shape}")
        return df

    def calculate_rul(self, df, dataset_type='train', rul_file_path=None):
        df2 = df.copy()
        if dataset_type == 'train':
            max_cycles = df2.groupby('engine_id')['cycle'].max().reset_index()
            max_cycles.columns = ['engine_id', 'max_cycle']
            df2 = df2.merge(max_cycles, on='engine_id')
            df2['RUL'] = df2['max_cycle'] - df2['cycle']
            df2.drop('max_cycle', axis=1, inplace=True)
        elif dataset_type == 'test' and rul_file_path and os.path.exists(rul_file_path):
            rul_true = pd.read_csv(rul_file_path, header=None, names=['RUL_true'])
            rul_true['engine_id'] = rul_true.index + 1
            last_cycles = df2.groupby('engine_id')['cycle'].max().reset_index()
            last_cycles = last_cycles.merge(rul_true, on='engine_id')
            last_cycles.columns = ['engine_id', 'cycle_last', 'RUL_true']
            df2 = df2.merge(last_cycles, on='engine_id')
            df2['RUL'] = df2['RUL_true'] + (df2['cycle_last'] - df2['cycle'])
            df2.drop(['cycle_last', 'RUL_true'], axis=1, inplace=True)
        else:
            df2['RUL'] = 0
        return df2

    def feature_engineering(self, df):
        df3 = df.copy()
        sensor_cols = [c for c in df3.columns if c.startswith('sensor_')]
        const = [c for c in sensor_cols if df3[c].std() == 0]
        df3.drop(columns=const, inplace=True)
        sensor_cols = [c for c in sensor_cols if c not in const]
        
        # Create comprehensive features to reach 66+
        for c in sensor_cols:
            grp = df3.groupby('engine_id')[c]
            df3[f'{c}_rollmean3'] = grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            df3[f'{c}_rollstd3'] = grp.rolling(3, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
            df3[f'{c}_rollmean5'] = grp.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
            df3[f'{c}_rollstd5'] = grp.rolling(5, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
            df3[f'{c}_diff1'] = grp.diff().fillna(0).reset_index(level=0, drop=True)
        
        # Operational settings
        op_cols = [c for c in df3.columns if c.startswith('op_setting_')]
        for c in op_cols:
            grp = df3.groupby('engine_id')[c]
            df3[f'{c}_rollmean3'] = grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
            df3[f'{c}_diff1'] = grp.diff().fillna(0).reset_index(level=0, drop=True)
        
        # Cycle features
        grp = df3.groupby('engine_id')['cycle']
        df3['cycle_rollmean'] = grp.rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df3['cycle_pct_max'] = df3.groupby('engine_id')['cycle'].transform(lambda x: x/x.max())
        
        print(f"Raw features after engineering: {len([c for c in df3.columns if c not in ['engine_id', 'cycle', 'RUL']])}")
        return df3

    def normalize_features(self, df, dataset_type):
        df4 = df.copy()
        drop_cols = ['engine_id', 'cycle', 'RUL']
        feat_path = os.path.join('data', 'processed', 'train', 'feature_columns_raw.txt')

        if dataset_type == 'test' and os.path.exists(feat_path):
            with open(feat_path) as f:
                trained = [l.strip() for l in f]
            print(f"Test: Loading {len(trained)} train features")
            for c in trained:
                if c not in df4.columns:
                    df4[c] = 0.0
            extra = [c for c in df4.columns if c not in trained + drop_cols]
            if extra:
                df4.drop(columns=extra, inplace=True)
            self.feature_cols = trained
        else:
            all_features = [c for c in df4.columns if c not in drop_cols]
            self.feature_cols = all_features[:66]  # Take first 66 to match model
            os.makedirs('data/processed/train', exist_ok=True)
            with open(feat_path, 'w') as f:
                for c in self.feature_cols:
                    f.write(c + '\n')
            print(f"Train: Selected exactly 66 features from {len(all_features)} available")

        print(f"Final features ({dataset_type}): {len(self.feature_cols)}")

        if self.scaler_type == 'standard':
            if dataset_type == 'train':
                self.scaler = StandardScaler()
                df4[self.feature_cols] = self.scaler.fit_transform(df4[self.feature_cols])
            else:
                scaler_path = os.path.join('data', 'processed', 'train', 'scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    df4[self.feature_cols] = self.scaler.transform(df4[self.feature_cols])
                else:
                    self.scaler = StandardScaler()
                    df4[self.feature_cols] = self.scaler.fit_transform(df4[self.feature_cols])
        return df4

    def generate_sequences(self, df):
        seqs, meta = [], []
        df_sorted = df.sort_values(['engine_id', 'cycle'])
        for eid, grp in df_sorted.groupby('engine_id'):
            arr = grp[self.feature_cols].values
            cycles = grp['cycle'].values
            ruls = grp['RUL'].values
            for i in range(self.window_size - 1, len(arr)):
                seqs.append(arr[i-self.window_size+1:i+1])
                meta.append({
                    'engine_id': eid,
                    'cycle': cycles[i],
                    'RUL': ruls[i],
                    'sequence_idx': len(seqs)-1
                })
        sequences = np.array(seqs)
        print(f"Generated sequences: {sequences.shape}")
        return sequences, pd.DataFrame(meta)

    def save_processed_data(self, seqs, meta_df, out_dir, dataset_type):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'rolling_window_sequences.npy'), seqs)
        meta_df.to_csv(os.path.join(out_dir, 'sequence_metadata_with_RUL.csv'), index=False)
        
        n_features = seqs.shape[2]
        print(f"✅ Saved {dataset_type}: {seqs.shape} ({n_features} features)")
        
        if dataset_type == 'train' and self.scaler is not None:
            train_dir = 'data/processed/train'
            os.makedirs(train_dir, exist_ok=True)
            joblib.dump(self.scaler, os.path.join(train_dir, 'scaler.pkl'))
            feature_columns = [f't{t}_f{f}' for t in range(self.window_size) for f in range(n_features)]
            with open(os.path.join(train_dir, 'feature_columns.txt'), 'w') as f:
                for c in feature_columns:
                    f.write(c + '\n')
            print(f"✅ Dashboard files saved for {n_features} features")

    def process_data(self, raw_path, dataset_type, rul_file, out_dir):
        df = self.load_data(raw_path, dataset_type)
        df = self.calculate_rul(df, dataset_type, rul_file)
        df = self.feature_engineering(df)
        df = self.normalize_features(df, dataset_type)
        seqs, meta_df = self.generate_sequences(df)
        self.save_processed_data(seqs, meta_df, out_dir, dataset_type)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--dataset_type', default='train', choices=['train','test'])
    parser.add_argument('--rul_file', default=None)
    parser.add_argument('--window_size', type=int, default=30)
    parser.add_argument('--scaler_type', default='standard', choices=['standard','minmax'])
    args = parser.parse_args()

    base_raw = os.path.join('data', 'raw', args.input_file)
    out_dir = os.path.join('data', 'processed', args.dataset_type)

    dp = DataPreprocessor(window_size=args.window_size, scaler_type=args.scaler_type)
    dp.process_data(base_raw, args.dataset_type, args.rul_file, out_dir)

if __name__ == '__main__':
    # Clean previous run
    import shutil
    if os.path.exists('data/processed'):
        shutil.rmtree('data/processed')
    
    dp = DataPreprocessor(window_size=30, scaler_type='standard')
    
    print("=== PROCESSING TRAIN DATA ===")
    dp.process_data(
        raw_path='data/raw/train_FD001.txt',
        dataset_type='train',
        rul_file=None,
        out_dir='data/processed/train'
    )
    
    print("\n=== PROCESSING TEST DATA ===")
    dp.process_data(
        raw_path='data/raw/test_FD001.txt',
        dataset_type='test',
        rul_file='data/raw/RUL_FD001.txt',
        out_dir='data/processed/test'
    )
    
    print("\n🎉 ✅ COMPLETE! Ready for BiLSTM + Streamlit!")
