import sys
import os
# Add parent directory to path to import from src_files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import tensorflow as tf
from src_files import load_helpers as lh
from src_files import signal_preprocess as sig_prep
from src_files import manipulate_dataset as md


def load_ptbxl_metadata(data_dir='/home/hart/thesis/ECG-algos/data/ptb-xl/', with_labels=False, labels_file=None):
    '''
    Loads the PTB-XL database metadata CSV file.

    Args:
        data_dir (str): Path to the PTB-XL dataset directory.
        with_labels (bool): If True, merge with processed labels metadata.
        labels_file (str, optional): Path to labels metadata CSV. If None and with_labels=True,
                                      uses 'ECG-XPLAIM/output/metadata/ptb-xl_labels_metadata.csv'

    Returns:
        pd.DataFrame: PTB-XL metadata dataframe.
    '''
    metadata_path = data_dir + 'ptbxl_database.csv'
    df = pd.read_csv(metadata_path)

    if with_labels:
        if labels_file is None:
            # Default path for processed labels
            labels_file = '/home/hart/thesis/ECG-algos/ECG-XPLAIM/output/metadata/ptb-xl_labels_metadata.csv'

        labels_df = pd.read_csv(labels_file)
        # Merge on ecg_id to combine demographics with labels
        df = df.merge(labels_df, on='ecg_id', how='inner', suffixes=('', '_labels'))

        # Use filename_hr from labels if it exists, otherwise keep original
        if 'filename_hr_labels' in df.columns:
            df['filename_hr'] = df['filename_hr_labels']
            df = df.drop('filename_hr_labels', axis=1)

    return df


def filter_by_sex(metadata_df, sex):
    '''
    Filters PTB-XL dataset by sex.

    Args:
        metadata_df (pd.DataFrame): PTB-XL metadata dataframe.
        sex (str or int): 'male'/'m'/1 for male, 'female'/'f'/0 for female.

    Returns:
        pd.DataFrame: Filtered dataframe containing only the specified sex.
    '''
    if isinstance(sex, str):
        sex = sex.lower()
        if sex in ['male', 'm']:
            sex_code = 1
        elif sex in ['female', 'f']:
            sex_code = 0
        else:
            raise ValueError("Sex must be 'male'/'m' or 'female'/'f'")
    else:
        sex_code = sex

    # Convert sex column to numeric to handle string values in CSV
    metadata_df = metadata_df.copy()
    metadata_df['sex'] = pd.to_numeric(metadata_df['sex'], errors='coerce')

    return metadata_df[metadata_df['sex'] == sex_code].copy()


def filter_by_age_range(metadata_df, min_age=None, max_age=None):
    '''
    Filters PTB-XL dataset by age range.

    Args:
        metadata_df (pd.DataFrame): PTB-XL metadata dataframe.
        min_age (int, optional): Minimum age (inclusive). None means no lower bound.
        max_age (int, optional): Maximum age (inclusive). None means no upper bound.

    Returns:
        pd.DataFrame: Filtered dataframe containing only records within the age range.
    '''
    filtered_df = metadata_df.copy()

    # Remove rows with missing age data
    filtered_df = filtered_df[filtered_df['age'].notna()]

    if min_age is not None:
        filtered_df = filtered_df[filtered_df['age'] >= min_age]
    if max_age is not None:
        filtered_df = filtered_df[filtered_df['age'] <= max_age]

    return filtered_df


def filter_by_age_group(metadata_df, age_group):
    '''
    Filters PTB-XL dataset by predefined age groups.

    Age groups:
        - 'pediatric': 0-17 years
        - 'young_adult': 18-35 years
        - 'adult': 36-64 years
        - 'elderly': 65+ years

    Args:
        metadata_df (pd.DataFrame): PTB-XL metadata dataframe.
        age_group (str): Age group name ('pediatric', 'young_adult', 'adult', 'elderly').

    Returns:
        pd.DataFrame: Filtered dataframe containing only the specified age group.
    '''
    age_groups = {
        'pediatric': (0, 17),
        'young_adult': (18, 35),
        'adult': (36, 64),
        'elderly': (65, None)
    }

    if age_group not in age_groups:
        raise ValueError(f"Age group must be one of {list(age_groups.keys())}")

    min_age, max_age = age_groups[age_group]
    return filter_by_age_range(metadata_df, min_age, max_age)


def filter_combined(metadata_df, sex=None, min_age=None, max_age=None, age_group=None):
    '''
    Applies multiple filters to the PTB-XL dataset.

    Args:
        metadata_df (pd.DataFrame): PTB-XL metadata dataframe.
        sex (str or int, optional): 'male'/'m'/1 or 'female'/'f'/0.
        min_age (int, optional): Minimum age (inclusive).
        max_age (int, optional): Maximum age (inclusive).
        age_group (str, optional): Predefined age group name.

    Returns:
        pd.DataFrame: Filtered dataframe.

    Note:
        If both age_group and min_age/max_age are specified, age_group takes precedence.
    '''
    filtered_df = metadata_df.copy()

    if sex is not None:
        filtered_df = filter_by_sex(filtered_df, sex)

    if age_group is not None:
        filtered_df = filter_by_age_group(filtered_df, age_group)
    elif min_age is not None or max_age is not None:
        filtered_df = filter_by_age_range(filtered_df, min_age, max_age)

    return filtered_df


def get_dataset_statistics(metadata_df):
    '''
    Computes demographic statistics for a PTB-XL dataset slice.

    Args:
        metadata_df (pd.DataFrame): PTB-XL metadata dataframe.

    Returns:
        dict: Dictionary containing demographic statistics.
    '''
    # Convert sex to numeric to handle string values
    sex_numeric = pd.to_numeric(metadata_df['sex'], errors='coerce')

    stats = {
        'total_records': len(metadata_df),
        'male_count': int((sex_numeric == 1).sum()),
        'female_count': int((sex_numeric == 0).sum()),
        'age_mean': float(metadata_df['age'].mean()) if len(metadata_df) > 0 else None,
        'age_std': float(metadata_df['age'].std()) if len(metadata_df) > 0 else None,
        'age_min': float(metadata_df['age'].min()) if len(metadata_df) > 0 else None,
        'age_max': float(metadata_df['age'].max()) if len(metadata_df) > 0 else None,
    }
    return stats


def get_file_paths(metadata_df, sampling_rate='hr'):
    '''
    Extracts file paths from filtered metadata.

    Args:
        metadata_df (pd.DataFrame): PTB-XL metadata dataframe.
        sampling_rate (str): 'hr' for high resolution (500Hz) or 'lr' for low resolution (100Hz).

    Returns:
        np.ndarray: Array of file paths for the filtered records.
    '''
    if sampling_rate == 'hr':
        return metadata_df['filename_hr'].values
    elif sampling_rate == 'lr':
        return metadata_df['filename_lr'].values
    else:
        raise ValueError("sampling_rate must be 'hr' or 'lr'")


def get_ecg_ids(metadata_df):
    '''
    Extracts ECG IDs from filtered metadata.

    Args:
        metadata_df (pd.DataFrame): PTB-XL metadata dataframe.

    Returns:
        np.ndarray: Array of ECG IDs for the filtered records.
    '''
    return metadata_df['ecg_id'].values


def save_filtered_metadata(metadata_df, output_path):
    '''
    Saves filtered metadata to CSV for use with existing TensorFlow dataset functions.

    Args:
        metadata_df (pd.DataFrame): Filtered PTB-XL metadata dataframe.
        output_path (str): Path where the filtered metadata CSV will be saved.

    Returns:
        str: Path to the saved metadata file.
    '''
    metadata_df.to_csv(output_path, index=False)
    return output_path


def tf_filtered_bal_dataset(
    data_input_dir,
    batch_size,
    n_samples_per_label,
    sex=None,
    min_age=None,
    max_age=None,
    age_group=None,
    shuffle_buffer=None,
    n_batches=None,
    repeat=False,
    preprocess_funcs={sig_prep.replace_nan: [0]}):
    '''
    Creates a balanced TensorFlow dataset from filtered PTB-XL data.
    Applies demographic filters (sex, age) before sampling.

    Args:
        data_input_dir (str): Path to input signal files (e.g., '/path/to/ptb-xl/').
        batch_size (int): Batch size.
        n_samples_per_label (dict): Number of samples per label, e.g., {'lqt': 1000, 'neg': 1000}.
        sex (str or int, optional): 'male'/'m'/1 or 'female'/'f'/0.
        min_age (int, optional): Minimum age (inclusive).
        max_age (int, optional): Maximum age (inclusive).
        age_group (str, optional): Predefined age group name.
        shuffle_buffer (int, optional): Shuffle buffer size.
        n_batches (int, optional): Number of batches to take.
        repeat (bool): Whether to repeat dataset.
        preprocess_funcs (dict): Dictionary of preprocessing functions to apply.

    Returns:
        tf.data.Dataset: A balanced, batched, preprocessed TensorFlow dataset.
    '''
    # Load metadata with labels
    labels_metadata = load_ptbxl_metadata(data_input_dir, with_labels=True)

    # Apply demographic filters
    if sex is not None or min_age is not None or max_age is not None or age_group is not None:
        labels_metadata = filter_combined(labels_metadata, sex, min_age, max_age, age_group)
        print(f"Filtered to {len(labels_metadata)} records based on demographics")

    # Prepare label list and file paths
    label_list = list(n_samples_per_label.keys())
    if 'neg' in label_list:
        label_list.remove('neg')

    all_file_paths = np.array(labels_metadata['filename_hr'])

    # Convert labels to a NumPy array for fast processing
    label_data = labels_metadata[label_list].values

    # Separate indices for each label
    label_indices = {label: np.where(label_data[:, i] == 1)[0] for i, label in enumerate(label_list)}
    neg_indices = np.where(label_data.sum(axis=1) == 0)[0]
    label_indices['neg'] = neg_indices

    # Check if we have enough samples
    for label, n_samples in n_samples_per_label.items():
        available = len(label_indices[label])
        if available < n_samples:
            print(f"Warning: Label '{label}' has only {available} samples, requested {n_samples}")

    # Aggregate indices for all labels, SHUFFLE, and remove duplicates
    aggregated_indices = []
    for label, n_samples in n_samples_per_label.items():
        available = len(label_indices[label])
        n_to_sample = min(n_samples, available)
        selected_indices = np.random.choice(label_indices[label], n_to_sample, replace=False)
        aggregated_indices.extend(selected_indices)

    # Remove duplicates and Shuffle
    aggregated_indices = list(set(aggregated_indices))
    np.random.shuffle(aggregated_indices)

    # Signal and label shapes
    signal_shape = lh.load_and_preprocess_signal(all_file_paths[0], ds_name='ptb-xl', data_input_dir=data_input_dir, preprocess_funcs=preprocess_funcs)[0].shape
    label_shape = (len(label_list), )

    def signal_generator():
        for idx in aggregated_indices:
            file_path = all_file_paths[idx]
            signal = lh.load_and_preprocess_signal(file_path, ds_name='ptb-xl', data_input_dir=data_input_dir, preprocess_funcs=preprocess_funcs)[0]
            label_values = np.zeros(label_shape, dtype=np.int32)

            # Assign label
            row_labels = label_data[idx]
            for i, value in enumerate(row_labels):
                if value == 1:
                    label_values[i] = 1

            yield signal, label_values

    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        signal_generator,
        output_signature=(
            tf.TensorSpec(shape=signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=label_shape, dtype=tf.int32)
        )
    )

    # Shuffle, batch, repeat, and prefetch
    dataset = md.shuf_bat_rep_pref(
        dataset,
        shuffle_buffer=shuffle_buffer,
        batch_size=batch_size,
        n_batches=n_batches,
        repeat=repeat
    )

    return dataset


def tf_filtered_dataset_from_indices(
    data_input_dir,
    metadata_df,
    indices,
    label_list,
    batch_size,
    shuffle_buffer=None,
    n_batches=None,
    repeat=False,
    preprocess_funcs={sig_prep.replace_nan: [0]}):
    '''
    Creates a TensorFlow dataset from specific indices in filtered metadata.
    Useful for creating custom train/val/test splits on filtered data.

    Args:
        data_input_dir (str): Path to input signal files.
        metadata_df (pd.DataFrame): Filtered PTB-XL metadata dataframe.
        indices (list or np.ndarray): Indices to include in the dataset.
        label_list (list): List of label column names.
        batch_size (int): Batch size.
        shuffle_buffer (int, optional): Shuffle buffer size.
        n_batches (int, optional): Number of batches to take.
        repeat (bool): Whether to repeat dataset.
        preprocess_funcs (dict): Dictionary of preprocessing functions to apply.

    Returns:
        tf.data.Dataset: A batched, preprocessed TensorFlow dataset.
    '''
    all_file_paths = metadata_df['filename_hr'].values
    label_data = metadata_df[label_list].values

    # Signal and label shapes
    signal_shape = lh.load_and_preprocess_signal(all_file_paths[0], ds_name='ptb-xl', data_input_dir=data_input_dir, preprocess_funcs=preprocess_funcs)[0].shape
    label_shape = (len(label_list), )

    def signal_generator():
        for idx in indices:
            file_path = all_file_paths[idx]
            signal = lh.load_and_preprocess_signal(file_path, ds_name='ptb-xl', data_input_dir=data_input_dir, preprocess_funcs=preprocess_funcs)[0]
            label_values = label_data[idx].astype(np.int32)
            yield signal, label_values

    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        signal_generator,
        output_signature=(
            tf.TensorSpec(shape=signal_shape, dtype=tf.float32),
            tf.TensorSpec(shape=label_shape, dtype=tf.int32)
        )
    )

    # Shuffle, batch, repeat, and prefetch
    dataset = md.shuf_bat_rep_pref(
        dataset,
        shuffle_buffer=shuffle_buffer,
        batch_size=batch_size,
        n_batches=n_batches,
        repeat=repeat
    )

    return dataset
