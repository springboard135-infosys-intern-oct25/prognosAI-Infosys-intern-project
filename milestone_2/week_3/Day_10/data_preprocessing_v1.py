# ============================================================================
# COMPLETE DATA PREPROCESSING PIPELINE - CMAPSS MULTI-DATASET (FD001-FD004)
# ============================================================================


# 1. IMPORTS
# ----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


# 2. DATA LOADING - MULTIPLE DATASETS
# ----------------------------------------------------------------------------
def load_multiple_cmapss_datasets(data_dir="C:/Users/win10/Desktop/Project_Oct25/prognosAI-Infosys-intern-project/data/raw",
                                  fd_list=[1, 2, 3, 4]):
    """
    Load multiple CMAPSS datasets (FD001-FD004) and merge them.
    
    Parameters:
    -----------
    data_dir : str or Path
        Directory containing the CMAPSS train files
    fd_list : list
        List of dataset IDs to load (default: [1, 2, 3, 4])
        
    Returns:
    --------
    df : pd.DataFrame
        Merged dataframe with all datasets
    datasets : dict
        Dictionary containing individual datasets
    """
    # Define column names: 3 operational settings + 21 sensors
    column_names = [
        "engine_id", "cycle", 
        "op_setting_1", "op_setting_2", "op_setting_3"
    ] + [f"sensor_{i}" for i in range(1, 22)]
    
    # Convert to Path object for cross-platform compatibility
    data_dir = Path(data_dir)
    
    # Load individual datasets
    datasets = {}
    for fd_id in fd_list:
        file_path = data_dir / f"train_FD00{fd_id}.txt"
        
        try:
            datasets[f"FD00{fd_id}"] = pd.read_csv(
                file_path,
                sep=r"\s+",
                header=None,
                names=column_names
            )
            # Add dataset identifier column
            datasets[f"FD00{fd_id}"]["dataset_id"] = f"FD00{fd_id}"
            print(f"✓ Loaded FD00{fd_id}: {datasets[f'FD00{fd_id}'].shape}")
        except FileNotFoundError:
            print(f"⚠ File not found: {file_path}")
            continue
    
    # Merge all datasets
    if len(datasets) == 0:
        raise ValueError("No datasets were loaded successfully!")
    
    df = pd.concat(datasets.values(), ignore_index=True)
    
    print(f"\n✓ Merged {len(datasets)} datasets")
    print(f"  Total shape: {df.shape}")
    
    return df, datasets


# 3. DATA PROFILING
# ----------------------------------------------------------------------------
def profile_data(df, verbose=True):
    """
    Generate comprehensive data profile report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print detailed information
        
    Returns:
    --------
    profile : dict
        Dictionary containing profiling information
    """
    profile = {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    if verbose:
        print("\n" + "="*70)
        print("DATA PROFILE REPORT")
        print("="*70)
        print(f"Shape: {profile['shape'][0]:,} rows × {profile['shape'][1]} columns")
        print(f"Memory Usage: {profile['memory_usage']:.2f} MB")
        print(f"Missing Values: {profile['missing_values']}")
        print(f"Duplicate Rows: {profile['duplicate_rows']}")
        
        print("\nData Types Distribution:")
        for dtype, count in profile['dtypes'].items():
            print(f"  {dtype}: {count} columns")
        
        print("\nDataset Distribution:")
        if 'dataset_id' in df.columns:
            print(df['dataset_id'].value_counts().sort_index())
        
        print("\n" + "-"*70)
        print("STATISTICAL SUMMARY (Sensors Only)")
        print("-"*70)
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        print(df[sensor_cols].describe().transpose()[['mean', 'std', 'min', 'max']])
    
    return profile


# 4. MISSING VALUE HANDLING
# ----------------------------------------------------------------------------
def handle_missing_values(df, method='ffill', group_by='dataset_id', verbose=True):
    """
    Handle missing values using specified imputation method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Imputation method: 'ffill', 'bfill', 'mean', 'median', 'drop'
    group_by : str or None
        Column to group by for grouped imputation (e.g., 'dataset_id')
    verbose : bool
        Print progress information
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with missing values handled
    """
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        if verbose:
            print("\n✓ No missing values detected")
        return df
    
    if verbose:
        print(f"\n⚠ Missing values detected: {missing_before}")
        missing_cols = df.isnull().sum()
        print("  Missing values per column:")
        print(missing_cols[missing_cols > 0])
    
    # Apply imputation method
    if method in ['ffill', 'bfill']:
        if group_by and group_by in df.columns:
            # Group-wise imputation to prevent data leakage
            df = df.groupby(group_by, sort=False).apply(
                lambda group: group.fillna(method=method).fillna(
                    group.fillna(method='bfill' if method == 'ffill' else 'ffill')
                )
            ).reset_index(drop=True)
        else:
            df = df.fillna(method=method).fillna(
                method='bfill' if method == 'ffill' else 'ffill'
            )
    elif method == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if group_by and group_by in df.columns:
            df[numeric_cols] = df.groupby(group_by, sort=False)[numeric_cols].transform(
                lambda x: x.fillna(x.mean())
            )
        else:
            df = df.fillna(df.mean(numeric_only=True))
    elif method == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if group_by and group_by in df.columns:
            df[numeric_cols] = df.groupby(group_by, sort=False)[numeric_cols].transform(
                lambda x: x.fillna(x.median())
            )
        else:
            df = df.fillna(df.median(numeric_only=True))
    elif method == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    missing_after = df.isnull().sum().sum()
    
    if verbose:
        print(f"✓ Missing values after handling: {missing_after}")
    
    return df


# 5. CONSTANT COLUMN DETECTION
# ----------------------------------------------------------------------------
def identify_constant_columns(df, threshold=0.0, verbose=True):
    """
    Identify columns with zero or near-zero variance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    threshold : float
        Standard deviation threshold (columns below this are constant)
    verbose : bool
        Print identified columns
        
    Returns:
    --------
    constant_cols : list
        List of constant column names
    """
    # Exclude categorical and identifier columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['engine_id', 'cycle']]
    
    std_values = df[numeric_cols].std()
    constant_cols = std_values[std_values <= threshold].index.tolist()
    
    if verbose and len(constant_cols) > 0:
        print(f"\n⚠ Identified {len(constant_cols)} constant columns:")
        for col in constant_cols:
            unique_vals = df[col].nunique()
            std_val = df[col].std()
            print(f"  - {col}: {unique_vals} unique value(s), std={std_val:.6f}")
    elif verbose:
        print("\n✓ No constant columns detected")
    
    return constant_cols


# 6. DUPLICATE DETECTION AND REMOVAL
# ----------------------------------------------------------------------------
def handle_duplicates(df, subset=None, remove=False, verbose=True):
    """
    Detect and optionally remove duplicate rows.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    subset : list or None
        Column names to consider for identifying duplicates
    remove : bool
        Whether to remove duplicates
    verbose : bool
        Print progress information
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with duplicates handled
    """
    duplicates = df.duplicated(subset=subset).sum()
    
    if verbose:
        if duplicates > 0:
            print(f"\n⚠ Detected {duplicates} duplicate rows")
        else:
            print("\n✓ No duplicate rows detected")
    
    if remove and duplicates > 0:
        df = df.drop_duplicates(subset=subset)
        if verbose:
            print(f"✓ Removed {duplicates} duplicate rows")
            print(f"  New shape: {df.shape}")
    
    return df


# 7. OUTLIER DETECTION
# ----------------------------------------------------------------------------
def detect_outliers(df, method='iqr', threshold=1.5, sensor_only=True, verbose=True):
    """
    Detect outliers using IQR or Z-score method.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        'iqr' or 'zscore'
    threshold : float
        IQR multiplier (default 1.5) or z-score threshold (default 1.5)
    sensor_only : bool
        Only check sensor columns for outliers
    verbose : bool
        Print outlier statistics
        
    Returns:
    --------
    outlier_info : dict
        Dictionary with outlier information per column
    """
    if sensor_only:
        numeric_cols = [col for col in df.columns if col.startswith('sensor_')]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    outlier_info = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z_scores > threshold).sum()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if outliers > 0:
            outlier_info[col] = outliers
    
    if verbose and len(outlier_info) > 0:
        print(f"\n⚠ Outliers detected ({method.upper()} method):")
        sorted_outliers = sorted(outlier_info.items(), key=lambda x: x[1], reverse=True)
        for col, count in sorted_outliers[:10]:  # Top 10
            pct = (count / len(df)) * 100
            print(f"  - {col}: {count} ({pct:.2f}%)")
        if len(sorted_outliers) > 10:
            print(f"  ... and {len(sorted_outliers) - 10} more columns")
    elif verbose:
        print("\n✓ No significant outliers detected")
    
    return outlier_info


# 8. DATA TYPE OPTIMIZATION
# ----------------------------------------------------------------------------
def validate_data_types(df, verbose=True):
    """
    Validate and optimize data types for memory efficiency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print optimization results
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with optimized data types
    """
    memory_before = df.memory_usage(deep=True).sum() / 1024**2
    
    # Optimize integer columns
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            else:
                df[col] = df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            else:
                df[col] = df[col].astype('int32')
    
    # Optimize float columns
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype('float32')
    
    memory_after = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        reduction = ((memory_before - memory_after) / memory_before * 100)
        print(f"\n✓ Data types optimized")
        print(f"  Memory before: {memory_before:.2f} MB")
        print(f"  Memory after: {memory_after:.2f} MB")
        print(f"  Reduction: {reduction:.1f}%")
    
    return df


# 9. AGGREGATE FEATURES (Engine-level statistics)
# -----------------------------------------------------------------------
def create_aggregate_features(df, verbose=True):
    """
    Create engine-level aggregate statistics for all sensors.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print progress information
        
    Returns:
    --------
    agg_df : pd.DataFrame
        Engine-level aggregate features (260 engines × 85 features)
    """
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    
    # Group by engine_id and calculate statistics
    agg_df = df.groupby('engine_id')[sensor_cols].agg(['mean', 'std', 'min', 'max'])
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    agg_df.reset_index(inplace=True)
    
    if verbose:
        print(f"\n✓ Aggregate features created")
        print(f"  Shape: {agg_df.shape}")
        print(f"  Engines: {len(agg_df)}")
        print(f"  Features per engine: {agg_df.shape[1] - 1}")  # Exclude engine_id
    
    return agg_df


# 10. ROLLING STATISTICS
# -----------------------------------------------------------------------
def create_rolling_features(df, window_size=5, verbose=True):
    """
    Create rolling mean and rolling std for each sensor per engine.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (must be sorted by engine_id and cycle)
    window_size : int
        Rolling window size in cycles (default: 5)
    verbose : bool
        Print progress information
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with added rolling features
    """
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    rolling_cols_added = 0
    
    for col in sensor_cols:
        # Create rolling mean and std within each engine group
        df[f"{col}_rollmean{window_size}"] = df.groupby('engine_id')[col].rolling(
            window=window_size, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        
        df[f"{col}_rollstd{window_size}"] = df.groupby('engine_id')[col].rolling(
            window=window_size, min_periods=1
        ).std().reset_index(level=0, drop=True)
        
        rolling_cols_added += 2
    
    if verbose:
        print(f"\n✓ Rolling features created")
        print(f"  Window size: {window_size} cycles")
        print(f"  Features added: {rolling_cols_added}")
        print(f"  New shape: {df.shape}")
        print(f"  Total rolling feature columns: {len(sensor_cols) * 2}")
    
    return df


# 11. FEATURE SCALING
# -----------------------------------------------------------------------
def scale_features(df, method='standard', exclude_cols=None, fit_scaler=True, 
                   scaler_obj=None, verbose=True):
    """
    Scale numerical features using StandardScaler or MinMaxScaler.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
    exclude_cols : list
        Columns to exclude from scaling
    fit_scaler : bool
        Whether to fit the scaler on this data
    scaler_obj : object
        Pre-fitted scaler object to use for transformation
    verbose : bool
        Print scaling information
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with scaled features
    scaler : object
        Fitted scaler object
    """
    if exclude_cols is None:
        exclude_cols = ['engine_id', 'cycle', 'dataset_id']
    
    # Select numeric columns to scale
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    # Initialize scaler
    if scaler_obj is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        scaler = scaler_obj
    
    # Fit and transform
    if fit_scaler:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    if verbose:
        print(f"\n✓ Features scaled using {method.upper()}")
        print(f"  Columns scaled: {len(cols_to_scale)}")
        print(f"  Scaling verification (first 3 features):")
        for col in cols_to_scale[:3]:
            print(f"    {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")
    
    return df, scaler


# 12. DATA SORTING AND ORDERING
# -----------------------------------------------------------------------
def sort_data_for_sequencing(df, verbose=True):
    """
    Sort data by engine_id and cycle to ensure correct temporal ordering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print progress information
        
    Returns:
    --------
    df : pd.DataFrame
        Sorted dataframe
    """
    df = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
    
    if verbose:
        print(f"\n✓ Data sorted by engine_id and cycle")
        print(f"  Shape maintained: {df.shape}")
    
    return df


# 13. FEATURE VALIDATION
# -----------------------------------------------------------------------
def validate_feature_quality(df, verbose=True):
    """
    Comprehensive validation of feature quality and integrity.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print validation results
        
    Returns:
    --------
    validation_report : dict
        Dictionary containing validation information
    """
    validation_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'inf_values': np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    }
    
    if verbose:
        print(f"\n✓ Feature quality validation")
        print(f"  Total rows: {validation_report['total_rows']:,}")
        print(f"  Total columns: {validation_report['total_columns']}")
        print(f"  Missing values: {validation_report['missing_values']}")
        print(f"  Duplicate rows: {validation_report['duplicate_rows']}")
        print(f"  Infinite values: {validation_report['inf_values']}")
        print(f"  Numeric columns: {validation_report['numeric_columns']}")
    
    return validation_report


# 14. REMOVE DATASET IDENTIFIER COLUMN
# -----------------------------------------------------------------------
def remove_dataset_identifier(df, verbose=True):
    """
    Remove the dataset_id column before saving (used for grouping during preprocessing).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print progress information
        
    Returns:
    --------
    df : pd.DataFrame
        Dataframe with dataset_id removed
    """
    if 'dataset_id' in df.columns:
        df = df.drop(columns=['dataset_id'])
        if verbose:
            print(f"\n✓ Removed 'dataset_id' column")
            print(f"  Shape after removal: {df.shape}")
    elif verbose:
        print(f"\n✓ 'dataset_id' column not found (already removed or not present)")
    
    return df


# 15. SAVE PROCESSING ARTIFACTS
# -----------------------------------------------------------------------
def save_preprocessing_artifacts(scaler, output_dir=".", verbose=True):
    """
    Save fitted scaler and preprocessing metadata for future use.
    
    Parameters:
    -----------
    scaler : object
        Fitted scaler object
    output_dir : str
        Directory to save artifacts
    verbose : bool
        Print progress information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    if verbose:
        print(f"\n✓ Preprocessing artifacts saved")
        print(f"  Scaler saved: {scaler_path}")


# 16. MAIN COMPLETE PREPROCESSING PIPELINE
# ============================================================================
def preprocess_cmapss_complete(
    data_dir="C:/Users/win10/Desktop/Project_Oct25/prognosAI-Infosys-intern-project/data/raw",
    fd_list=[1, 2, 3, 4],
    output_path="cmapss_preprocessed_complete.csv",
    output_dir=".",
    remove_constant_cols=True,
    remove_duplicates=False,
    optimize_dtypes=True,
    create_rolling=True,
    rolling_window=5,
    scale_sensors=True,
    scaling_method='standard',
    verbose=True):
    """
    Complete end-to-end preprocessing pipeline with feature engineering.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CMAPSS train files
    fd_list : list
        List of dataset IDs to load
    output_path : str
        Path to save final preprocessed data
    output_dir : str
        Directory to save artifacts
    remove_constant_cols : bool
        Remove zero-variance columns
    remove_duplicates : bool
        Remove duplicate rows
    optimize_dtypes : bool
        Optimize data types for memory efficiency
    create_rolling : bool
        Create rolling window features
    rolling_window : int
        Rolling window size (cycles)
    scale_sensors : bool
        Scale sensor features
    scaling_method : str
        'standard' or 'minmax'
    verbose : bool
        Print detailed progress
        
    Returns:
    --------
    df_processed : pd.DataFrame
        Fully preprocessed dataframe
    scaler : object
        Fitted scaler object
    preprocessing_summary : dict
        Summary of preprocessing steps
    """
    if verbose:
        print("\n" + "="*80)
        print("COMPLETE DATA PREPROCESSING PIPELINE - CMAPSS MULTI-DATASET")
        print("="*80)
    
    preprocessing_summary = {
        'steps_completed': [],
        'data_shapes': {},
        'features_created': {},
        'memory_info': {}
    }
    
    # Step 1: Load data
    if verbose:
        print("\n[1/12] LOADING DATASETS...")
    df, datasets = load_multiple_cmapss_datasets(data_dir, fd_list)
    preprocessing_summary['data_shapes']['after_loading'] = df.shape
    preprocessing_summary['steps_completed'].append('Data Loading')
    
    # Step 2: Profile data
    if verbose:
        print("\n[2/12] PROFILING DATA...")
    profile = profile_data(df, verbose=verbose)
    preprocessing_summary['memory_info']['before_cleaning'] = profile['memory_usage']
    preprocessing_summary['steps_completed'].append('Data Profiling')
    
    # Step 3: Sort data
    if verbose:
        print("\n[3/12] SORTING DATA...")
    df = sort_data_for_sequencing(df, verbose=verbose)
    preprocessing_summary['steps_completed'].append('Data Sorting')
    
    # Step 4: Handle missing values
    if verbose:
        print("\n[4/12] HANDLING MISSING VALUES...")
    df = handle_missing_values(df, method='ffill', group_by='dataset_id', verbose=verbose)
    preprocessing_summary['steps_completed'].append('Missing Value Imputation')
    
    # Step 5: Identify constant columns
    if verbose:
        print("\n[5/12] DETECTING CONSTANT COLUMNS...")
    constant_cols = identify_constant_columns(df, threshold=0.0, verbose=verbose)
    if remove_constant_cols and len(constant_cols) > 0:
        df = df.drop(columns=constant_cols)
        preprocessing_summary['steps_completed'].append(f'Removed {len(constant_cols)} constant columns')
    
    # Step 6: Handle duplicates
    if verbose:
        print("\n[6/12] HANDLING DUPLICATES...")
    df = handle_duplicates(df, remove=remove_duplicates, verbose=verbose)
    preprocessing_summary['steps_completed'].append('Duplicate Detection/Removal')
    
    # Step 7: Detect outliers
    if verbose:
        print("\n[7/12] DETECTING OUTLIERS...")
    outlier_info = detect_outliers(df, method='iqr', threshold=1.5, sensor_only=True, verbose=verbose)
    preprocessing_summary['features_created']['outliers_detected'] = sum(outlier_info.values())
    preprocessing_summary['steps_completed'].append('Outlier Detection')
    
    # Step 8: Create rolling features
    scaler = None
    if create_rolling:
        if verbose:
            print(f"\n[8/12] CREATING ROLLING FEATURES (window={rolling_window})...")
        df = create_rolling_features(df, window_size=rolling_window, verbose=verbose)
        preprocessing_summary['features_created']['rolling_features'] = len([c for c in df.columns if 'roll' in c])
        preprocessing_summary['steps_completed'].append(f'Rolling Features (window={rolling_window})')
        
        # Handle NaN from rolling calculations
        if verbose:
            print("\n  Handling NaN values from rolling calculations...")
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        missing_after = df.isnull().sum().sum()
        if verbose:
            print(f"  ✓ Dropped {len(df) - len(df.dropna())} rows with NaN")
    
    # Step 9: Optimize data types
    if verbose:
        print("\n[9/12] OPTIMIZING DATA TYPES...")
    if optimize_dtypes:
        df = validate_data_types(df, verbose=verbose)
        preprocessing_summary['steps_completed'].append('Data Type Optimization')
    
    # Step 10: Scale features
    if verbose:
        print("\n[10/12] SCALING FEATURES...")
    if scale_sensors:
        df, scaler = scale_features(df, method=scaling_method, verbose=verbose)
        preprocessing_summary['steps_completed'].append(f'Feature Scaling ({scaling_method})')
        save_preprocessing_artifacts(scaler, output_dir, verbose=False)
    
    # Step 11: Validate features
    if verbose:
        print("\n[11/12] FINAL VALIDATION...")
    validation = validate_feature_quality(df, verbose=verbose)
    preprocessing_summary['data_shapes']['before_dataset_removal'] = df.shape
    preprocessing_summary['data_shapes']['feature_validation'] = validation
    preprocessing_summary['steps_completed'].append('Feature Validation')
    
    # Step 12: Remove dataset_id column and save
    if verbose:
        print("\n[12/12] REMOVING DATASET IDENTIFIER & SAVING...")
    df = remove_dataset_identifier(df, verbose=verbose)
    preprocessing_summary['data_shapes']['after_preprocessing'] = df.shape
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    preprocessing_summary['output_file'] = str(output_path)
    if verbose:
        print(f"\n✓ Preprocessed data saved: {output_path}")
    
    # Final summary
    if verbose:
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE - FINAL SUMMARY")
        print("="*80)
        print(f"Total rows: {df.shape[0]:,}")
        print(f"Total features: {df.shape[1]}")
        print(f"Sensor columns: {len([c for c in df.columns if c.startswith('sensor_') and 'roll' not in c])}")
        print(f"Rolling feature columns: {preprocessing_summary['features_created'].get('rolling_features', 0)}")
        print(f"\nFinal column breakdown:")
        print(f"  - engine_id: 1")
        print(f"  - cycle: 1")
        print(f"  - Operational settings: 3")
        print(f"  - Raw sensors: {len([c for c in df.columns if c.startswith('sensor_') and 'roll' not in c])}")
        print(f"  - Rolling features: {preprocessing_summary['features_created'].get('rolling_features', 0)}")
        print(f"\nSteps completed:")
        for i, step in enumerate(preprocessing_summary['steps_completed'], 1):
            print(f"  {i}. {step}")
    
    return df, scaler, preprocessing_summary


# 17. USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Configure paths
    DATA_DIR = "C:/Users/win10/Desktop/Project_Oct25/prognosAI-Infosys-intern-project/data/raw"
    OUTPUT_PATH = "C:/Users/win10/Desktop/Project_Oct25/prognosAI-Infosys-intern-project/data/processed/cmapss_preprocessed.csv"
    OUTPUT_DIR = "."
    
    # Run complete preprocessing pipeline
    df_processed, scaler, summary = preprocess_cmapss_complete(
        data_dir=DATA_DIR,
        fd_list=[1, 2, 3, 4],
        output_path=OUTPUT_PATH,
        output_dir=OUTPUT_DIR,
        remove_constant_cols=True,
        remove_duplicates=False,
        optimize_dtypes=True,
        create_rolling=True,
        rolling_window=5,
        scale_sensors=True,
        scaling_method='standard',
        verbose=True
    )
    
    # Display results
    print("\n" + "-"*80)
    print("PREPROCESSED DATA SAMPLE (First 10 Rows)")
    print("-"*80)
    print(df_processed.head(10))
    
    print("\n" + "-"*80)
    print("FINAL FEATURE COLUMNS")
    print("-"*80)
    print(f"Total: {len(df_processed.columns)} columns")
    print(f"\nColumn breakdown:")
    print(f"  - Identifiers: engine_id, cycle")
    print(f"  - Operational settings: 3 (op_setting_1, op_setting_2, op_setting_3)")
    print(f"  - Raw sensors: {len([c for c in df_processed.columns if c.startswith('sensor_') and 'roll' not in c])}")
    print(f"  - Rolling features: {len([c for c in df_processed.columns if 'roll' in c])}")
    
    print("\n" + "-"*80)
    print("VERIFY dataset_id REMOVAL")
    print("-"*80)
    if 'dataset_id' in df_processed.columns:
        print("❌ WARNING: 'dataset_id' column still present!")
    else:
        print("✓ CONFIRMED: 'dataset_id' column successfully removed")
    
    print("\n" + "-"*80)
    print("SCALING VERIFICATION (Sample Features)")
    print("-"*80)
    sensor_sample = [c for c in df_processed.columns if c.startswith('sensor_') and 'roll' not in c][:3]
    print(df_processed[sensor_sample].describe().transpose()[['mean', 'std', 'min', 'max']])
    
    print("\n" + "-"*80)
    print("FINAL DATAFRAME INFO")
    print("-"*80)
    print(df_processed.info())
