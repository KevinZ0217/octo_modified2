import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow_datasets as tfds
import tensorflow as tf

# Load the original dataset using the builder
builder = tfds.builder("cmu_stretch:0.1.0", data_dir="/mnt/cube/datasets/X-embodiment")

# builder.download_and_prepare()

# Load the dataset from the builder
dataset = builder.as_dataset(split='train')

# Process the dataset (e.g., filter out some data points)
def filter_fn(example):
    # Your filtering logic here (return True for data to keep, False otherwise)
    return example['label'] == 1  # Example condition: keep only examples with label == 1

# filtered_dataset = dataset.filter(filter_fn)
filtered_dataset = dataset
# Optionally apply more transformations here, e.g., map, etc.

# Save the filtered dataset back using the original builder
class ModifiedDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    VERSION = builder.VERSION  # Keep the original version

    @property
    def name(self):
        # Return the same dataset name as the original 'cmu_stretch'
        return "cmu_stretch"


    def _info(self):
        return builder.info  # Use the original dataset info

    def _split_generators(self, dl_manager):
        # Define the split (you can modify this based on your filtered data)
        return {
            'train': self._generate_examples(filtered_dataset),
        }

    def _generate_examples(self, dataset):
        # Generator function that yields examples from your filtered dataset
        for i, example in enumerate(tfds.as_numpy(dataset)):  # Convert to numpy for storage
            print(example)
            yield i, example

# Instantiate and save the modified dataset
modified_builder = ModifiedDatasetBuilder(data_dir="/mnt/cube/datasets/x-embodiment-test/cmu_stretch")  # Use the original data directory
print("here")
modified_builder.download_and_prepare()

# You can now load the filtered dataset again to verify
filtered_dataset_saved = modified_builder.as_dataset(split='train')
