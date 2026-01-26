'''
Examples demonstrating how to use filter_dataset.py with manipulate_dataset.py
to create filtered TensorFlow datasets from PTB-XL data.
'''

import sys
import os
# Add parent directory to path to import from src_files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import filter_dataset as fd
from src_files import manipulate_dataset as md
from src_files import signal_preprocess as sig_prep


# Example 1: Basic demographic filtering and statistics
print("=" * 60)
print("Example 1: Filter by demographics and get statistics")
print("=" * 60)

# Load full metadata
metadata = fd.load_ptbxl_metadata('/home/hart/thesis/ECG-algos/data/ptb-xl/')

# Get only female patients
females = fd.filter_by_sex(metadata, 'female')
print(f"\nFemale patients: {len(females)}")

# Get elderly male patients
elderly_males = fd.filter_combined(metadata, sex='male', age_group='elderly')
print(f"Elderly male patients: {len(elderly_males)}")

# Get statistics
stats = fd.get_dataset_statistics(elderly_males)
print(f"\nStatistics for elderly males:")
print(f"  Total: {stats['total_records']}")
if stats['age_mean'] is not None:
    print(f"  Age: {stats['age_mean']:.1f} ± {stats['age_std']:.1f} years")
    print(f"  Range: {stats['age_min']:.0f}-{stats['age_max']:.0f} years")
else:
    print("  No records to compute statistics")


# Example 2: Create a filtered TensorFlow dataset directly
print("\n" + "=" * 60)
print("Example 2: Create filtered TensorFlow dataset")
print("=" * 60)

# Create a balanced dataset with only female patients
# Using actual label columns from PTB-XL: afib, ami, lqt, etc.
female_dataset = fd.tf_filtered_bal_dataset(
    data_input_dir='/home/hart/thesis/ECG-algos/data/ptb-xl/',
    batch_size=32,
    n_samples_per_label={'afib': 100, 'ami': 100, 'lqt': 100},
    sex='female',
    shuffle_buffer=1000,
    preprocess_funcs={sig_prep.replace_nan: [0]}
)

print("\nCreated filtered female dataset")
print("This dataset contains only female patients with balanced labels")


# Example 3: Create datasets for different age groups
print("\n" + "=" * 60)
print("Example 3: Create datasets for different age groups")
print("=" * 60)

# Young adults dataset (18-35)
young_dataset = fd.tf_filtered_bal_dataset(
    data_input_dir='/home/hart/thesis/ECG-algos/data/ptb-xl/',
    batch_size=32,
    n_samples_per_label={'afib': 50, 'ami': 50},
    age_group='young_adult',
    shuffle_buffer=500
)

# Elderly dataset (65+)
elderly_dataset = fd.tf_filtered_bal_dataset(
    data_input_dir='/home/hart/thesis/ECG-algos/data/ptb-xl/',
    batch_size=32,
    n_samples_per_label={'afib': 100, 'ami': 100},
    age_group='elderly',
    shuffle_buffer=500
)

print("\nCreated age-stratified datasets for young adults and elderly")


# Example 4: Custom age range
print("\n" + "=" * 60)
print("Example 4: Custom age range filtering")
print("=" * 60)

# Middle-aged patients (40-60 years)
middle_aged_dataset = fd.tf_filtered_bal_dataset(
    data_input_dir='/home/hart/thesis/ECG-algos/data/ptb-xl/',
    batch_size=32,
    n_samples_per_label={'afib': 100, 'ami': 100},
    min_age=40,
    max_age=60,
    shuffle_buffer=800
)

print("\nCreated dataset for middle-aged patients (40-60 years)")


# Example 5: Multiple filters combined
print("\n" + "=" * 60)
print("Example 5: Combine multiple demographic filters")
print("=" * 60)

# Young adult females
young_female_dataset = fd.tf_filtered_bal_dataset(
    data_input_dir='/home/hart/thesis/ECG-algos/data/ptb-xl/',
    batch_size=32,
    n_samples_per_label={'afib': 50, 'ami': 50},
    sex='female',
    age_group='young_adult',
    shuffle_buffer=400
)

print("\nCreated dataset for young adult females (18-35, female)")


# Example 6: Save filtered metadata for later use
print("\n" + "=" * 60)
print("Example 6: Save filtered metadata and use with original functions")
print("=" * 60)

# Filter metadata
elderly_females = fd.filter_combined(
    metadata,
    sex='female',
    age_group='elderly'
)

# Save to temporary file
filtered_path = '/tmp/elderly_females_metadata.csv'
fd.save_filtered_metadata(elderly_females, filtered_path)
print(f"\nSaved filtered metadata to {filtered_path}")
print(f"Contains {len(elderly_females)} records")

# Now you can manually use this with other processing scripts


# Example 7: Create train/val/test splits from filtered data
print("\n" + "=" * 60)
print("Example 7: Create train/val/test splits on filtered data")
print("=" * 60)

# First create the full filtered dataset
full_dataset = fd.tf_filtered_bal_dataset(
    data_input_dir='/home/hart/thesis/ECG-algos/data/ptb-xl/',
    batch_size=32,
    n_samples_per_label={'afib': 200, 'ami': 200},
    sex='male',
    age_group='adult',
    shuffle_buffer=2000
)

# Split into train/val/test
train_dataset, val_dataset, test_dataset = md.tf_train_val_test_sets(
    full_dataset,
    n_train=50,   # 50 batches for training
    n_val=10,     # 10 batches for validation
    n_test=10,    # 10 batches for testing
    train_repeat=True,
    val_repeat=False,
    test_repeat=False
)

print("\nCreated train/val/test splits from filtered dataset")
print("Train: 50 batches (repeated)")
print("Val: 10 batches")
print("Test: 10 batches")


# Example 8: Compare datasets across demographics
print("\n" + "=" * 60)
print("Example 8: Compare statistics across different groups")
print("=" * 60)

groups = {
    'All': metadata,
    'Male': fd.filter_by_sex(metadata, 'male'),
    'Female': fd.filter_by_sex(metadata, 'female'),
    'Young Adults': fd.filter_by_age_group(metadata, 'young_adult'),
    'Elderly': fd.filter_by_age_group(metadata, 'elderly')
}

print("\nDemographic comparison:")
print(f"{'Group':<15} {'Count':>8} {'Age Mean':>10} {'Male %':>10}")
print("-" * 45)

for name, df in groups.items():
    stats = fd.get_dataset_statistics(df)
    male_pct = 100 * stats['male_count'] / stats['total_records'] if stats['total_records'] > 0 else 0
    age_mean = stats['age_mean'] if stats['age_mean'] is not None else 0
    print(f"{name:<15} {stats['total_records']:>8} {age_mean:>10.1f} {male_pct:>9.1f}%")


print("\n" + "=" * 60)
print("All examples completed!")
print("=" * 60)
