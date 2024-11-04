from functools import partial
import inspect
import json
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from octo.data import obs_transforms, traj_transforms
from octo.data.utils import goal_relabeling, task_augmentation
from octo.data.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    NormalizationType,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
)

import matplotlib.pyplot as plt


# Filter out the high total variation trajectory
def calculate_total_variation(traj):
    action_diff = tf.abs(tf.experimental.numpy.diff(traj['action'][:, :3], axis=0))
    total_variation = tf.reduce_sum(action_diff, axis=0)
    return total_variation

def get_total_variation_threshold(variations):
    return np.percentile(variations, 60)

def filter_total_variation(dataset, variations_by_task):
    def filter_fn(traj):
        task =  tf.cast(traj["task"]["language_instruction"], tf.string)
        print(task)
        total_variation = calculate_total_variation(traj)
        threshold = variations_by_task[task]
        return tf.math.reduce_all(total_variation <= threshold)

    # dataset = dataset.filter(
    #         lambda x: tf.math.reduce_all(calculate_total_variation(x) <= variations_by_task[x['task']['language_instruction'].numpy().decode('utf-8')])
    #     )

    return dataset.filter(filter_fn)

# def filter_total_variation(dataset, variations_by_task):

#     # Initialize an empty list to store filtered trajectories
#     filtered_trajectories = []
    
#     # Iterate over the dataset manually
#     for traj in dataset:
#         # Access the task (language instruction)
#         task = np.array(traj["task"]["language_instruction"])[0]
#         print(task)
#         # Calculate total variation or any other metric
#         total_variation = calculate_total_variation(traj)
        
#         # Apply your filtering condition
#         threshold = variations_by_task[task]
#         if tf.reduce_all(total_variation <= threshold):
#             # If the condition is met, add the trajectory to the filtered list
#             filtered_trajectories.append(traj)
    
#     # Rebuild the dataset from the filtered list
#     filtered_dataset = tf.data.Dataset.from_generator(
#         lambda: iter(filtered_trajectories),
#         output_signature=dataset.element_spec
#     )

#     return filtered_dataset

def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of
    "relabeling" (e.g. filtering, chunking, adding goals, dropping keys). Transforms that happen in this
    function should have the following properties:

    - They require access to an entire trajectory (i.e. they cannot be applied in a frame-wise manner).
    - They are generally not CPU-intensive, mostly involving moving and copying data.
    - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        future_action_window_size (int, optional): The number of future actions beyond window_size to include
            in the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be subsampled to
            this length (after goal relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
            function.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError(
                "skip_unlabeled=True but dataset does not have language labels."
            )
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )
        
    #check the distribution of actions before filtering
    final_action_list_before = np.empty((0,7))
    for traj in dataset:
        final_action_list_before = np.vstack((final_action_list_before,traj["action"]))
    # print(np.array(final_action_list_before).shape)
    stds = np.std(final_action_list_before,axis=0)
    means = np.mean(final_action_list_before,axis=0)
    # fig, axes = plt.subplots(7, 1, figsize=(10, 14))
    upper_bound_list = []
    lower_bound_list = []
    for i in range(7):
        # axes[i].hist(final_action_list_before[:, i], bins=1000, alpha=0.7, color='blue')
        # axes[i].set_title(f'Distribution of Dimension before filtering {i+1}')
        # axes[i].set_xlabel('Value')
        # axes[i].set_ylabel('Frequency')

        mean = means[i]
        std_dev = stds[i]
        # print(mean, std_dev)
        lower_bound = mean - 4.5 * std_dev
        upper_bound = mean + 4.5 * std_dev
        upper_bound_list.append(upper_bound)
        lower_bound_list.append(lower_bound)
        
        # axes[i].axvline(mean, color='red', linestyle='dashed', linewidth=1)
        # axes[i].axvline(lower_bound, color='green', linestyle='dashed', linewidth=1)
        # axes[i].axvline(upper_bound, color='green', linestyle='dashed', linewidth=1)

    # plt.tight_layout()
    # plt.show()
    outlier_counter = 0
    traj_counter = 0
    for traj in dataset:
        traj_counter+=1
        for i in range(7):
            # print(traj["action"][:,i])
            if tf.reduce_any(tf.greater(traj["action"][:,i], upper_bound_list[i])) or tf.reduce_any(tf.less(traj["action"][:,i] , lower_bound_list[i])):
                outlier_counter+=1
                break
    # print(means,stds,upper_bound_list,lower_bound_list)
    print("upper bound:",upper_bound_list)            
    print("lower bound:",lower_bound_list)   
    
    
    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    #check the distribution of actions
    print(np.array(final_action_list_before).shape)
    # fig, axes = plt.subplots(7, 1, figsize=(10, 14))

    # for i in range(7):
    #     axes[i].hist(final_action_list_before[:, i], bins=20, alpha=0.7, color='blue')
    #     axes[i].set_title(f'Distribution of Dimension {i+1}')
    #     axes[i].set_xlabel('Value')
    #     axes[i].set_ylabel('Frequency')
    
    # plt.tight_layout()
    # plt.show()

    
    
    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )
    #filter out high total variation
    variations_by_task = {}
    all_variations = {}

    # for traj in dataset:
    #     task = np.array(traj['task']['language_instruction'])[0]
    #     # print(traj)
    #     variation = np.linalg.norm(calculate_total_variation(traj))
    #     if task not in all_variations:
    #         all_variations[task] = []
    #     all_variations[task].append(variation)

    # for task, variations in all_variations.items():
    #     variations_by_task[task] = get_total_variation_threshold(variations)

    # # Filter based on the top 60% of total variations
    # print(variations_by_task)
    # dataset = filter_total_variation(dataset, variations_by_task)
    
    # marks which entires of the observation and task dicts are padding
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions, giving them a new axis at index 1 of size `window_size` and
    # `window_size + future_action_window_size`, respectively
    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    return dataset,outlier_counter,traj_counter


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a frame level. These transforms are usually more
    CPU-intensive, (e.g. decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
            images.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
        # task is not chunked -- apply fn directly
        frame["task"] = fn(frame["task"])
        # observation is chunked -- apply fn along first axis
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    # decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )

    if train:
        # augment all images with the same seed, skipping padding images
        def aug(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(
                obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs
            )
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    state_obs_keys: Sequence[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[Sequence[bool]] = None,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation" and "action". "observation"
    should be a dictionary containing some number of additional keys, which will be extracted into an even
    more standardized format according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in
    place of an old name to insert padding. For example, if after `standardize_fn`, your "observation" dict
    has RGB images called "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary":
    None, "wrist": "wrist"}`, then the resulting dataset will have an "observation" dict containing the keys
    "image_primary", "image_secondary", and "image_wrist", where "image_primary" corresponds to "workspace",
    "image_secondary" is a padding image, and "image_wrist" corresponds to "wrist".

    `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which
    will be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be
    inserted for each None entry.

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will
    contain the key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset,
            since one file usually contains many trajectories!).
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to
            extract from the "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in
            image_obs_keys.items()}`. If a value of `old` is None, inserts a padding image instead (empty
            string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from
            the "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding (zero) for
            each None entry.
        language_key (str, optional): If provided, the "task" dict will contain the key
            "language_instruction", extracted from `traj[language_key]`.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        absolute_action_mask (Sequence[bool], optional): By default, all action dimensions are assumed to be
            relative. This is important for when `future_action_window_size > 0`: actions that are taken
            from beyond the end of the trajectory (or beyond the goal timestep when goal relabeling is used)
            need to be made "neutral" to indicate that the task has been completed. For relative actions,
            "neutral" means zero, but for absolute actions, "neutral" means repeating the last valid action.
            This mask, if provided, indicates which action dimensions are absolute.
        action_normalization_mask (Sequence[bool], optional): If provided, indicates which action dimensions
            should be normalized. For example, you might not want to normalize the gripper action dimension if
            it's always exactly 0 or 1. By default, all action dimensions are normalized.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)

    def restructure(traj):
        # apply a standardization function, if provided
        # TODO: figure out the standardize function 
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. "
                "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                    if key is None
                    else tf.cast(old_obs[key], tf.float32)
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, "
                    "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)
            
            
            
        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj
    
    builder = tfds.builder(f"{name}:0.1.0", data_dir=data_dir)

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure, num_parallel_calls)
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
            save_dir=builder.data_dir,
        )
    print(dataset_statistics)
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if (
            len(action_normalization_mask)
            != dataset_statistics["action"]["mean"].shape[-1]
        ):
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # construct the dataset
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )
    #count the outliers
    dataset = dataset.traj_map(restructure, num_parallel_calls)
    whole_action_list = np.empty((0,7))
    traj_counter = 0
    for traj in dataset:
        whole_action_list = np.vstack((whole_action_list,traj["action"]))
        traj_counter+=1
    fig, axes = plt.subplots(7, 1, figsize=(10, 14))
    for i in range(7):
        axes[i].hist(whole_action_list[:, i], bins=200, alpha=0.7, color='blue')
        axes[i].set_title(f'Distribution of Dimension {i+1}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

    
    # print(np.array(final_action_list_before).shape)
    stds = np.std(whole_action_list,axis=0)
    means = np.mean(whole_action_list,axis=0)
    print("std mean list length:",stds.shape, means.shape)
    upper_bound_list = []
    lower_bound_list = []
    for i in range(7):

        mean = means[i]
        std_dev = stds[i]
        lower_bound = mean - 4.5 * std_dev
        upper_bound = mean + 4.5 * std_dev
        upper_bound_list.append(upper_bound)
        lower_bound_list.append(lower_bound)
    
    # outlier_counter = 0
    # traj_counter = 0
    # for traj in dataset:
    #     traj_counter+=1
    #     for i in range(7):
    #         # print(traj["action"][:,i])
    #         if tf.reduce_any(tf.greater(traj["action"][:,i], upper_bound_list[i])) or tf.reduce_any(tf.less(traj["action"][:,i] , lower_bound_list[i])):
    #             outlier_counter+=1
    #             break
    print("later:",lower_bound_list,upper_bound_list)
    outlier_dataset = dataset
    # for i in range(7):
    #     new_dataset = new_dataset.filter(lambda traj:
    #                 tf.reduce_any(tf.less_equal(traj["action"][:, i], lower_bound_list[i])) or
    #                 tf.reduce_any(tf.greater_equal(traj["action"][:, i], upper_bound_list[i]))
    #             )
    outlier_dataset = outlier_dataset.filter(lambda traj:

                        tf.logical_or(
                            tf.reduce_any(tf.reduce_any(tf.less(traj["action"], lower_bound_list),axis=1)),
                            tf.reduce_any(tf.reduce_any(tf.greater(traj["action"], upper_bound_list), axis=1))
                        )
    )
    
    outlier_counter = 0
    for traj in outlier_dataset:
        outlier_counter+=1
    # fig, axes = plt.subplots(7, 1, figsize=(10, 14))
    # for i in range(7):
    #     axes[i].hist(whole_action_list[:, i], bins=200, alpha=0.7, color='blue')
    #     axes[i].set_title(f'Distribution of Dimensions {i+1}')
    #     axes[i].set_xlabel('Values')
    #     axes[i].set_ylabel('Frequencys')
    
    # plt.tight_layout()
    # plt.show()






    dataset = dataset.traj_map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )
    print("dataset type:",type(dataset))

    return dataset, dataset_statistics, outlier_counter, traj_counter


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics, outlier_counter1, traj_counter1 = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )

    dataset,outlier_counter2,traj_counter2 = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    whole_action_list = np.empty((0,7))
    for traj in dataset:
        action = np.array(traj["action"])
        whole_action_list = np.vstack((whole_action_list,np.squeeze(action)))
    
    # print(np.array(final_action_list_before).shape)
    stds = np.std(whole_action_list,axis=0)
    means = np.mean(whole_action_list,axis=0)
    upper_bound_list = []
    lower_bound_list = []
    for i in range(7):
        mean = means[i]
        std_dev = stds[i]
        lower_bound = mean - 4.5 * std_dev
        upper_bound = mean + 4.5 * std_dev
        upper_bound_list.append(upper_bound)
        lower_bound_list.append(lower_bound)

    outlier_dataset = dataset

    whole_data_counter = 0
    for traj in dataset:
        whole_data_counter += 1

    outlier_dataset = outlier_dataset.filter(lambda traj:

                        tf.logical_or(
                            tf.reduce_any(tf.reduce_any(tf.less(traj["action"], lower_bound_list),axis=1)),
                            tf.reduce_any(tf.reduce_any(tf.greater(traj["action"], upper_bound_list), axis=1))
                        )
    )
    dataset = dataset.filter(lambda traj:

                        tf.logical_and(
                            tf.reduce_all(tf.reduce_all(tf.less_equal(traj["action"], upper_bound_list),axis=1)),
                            tf.reduce_all(tf.reduce_all(tf.greater_equal(traj["action"],lower_bound_list), axis=1))
                        )
    )


    final_outlier_counter = 0
    final_data_counter = 0
    for traj in outlier_dataset:
        final_outlier_counter+=1
    
    for traj in dataset:
        final_data_counter+=1
    print("outlier final check:",final_outlier_counter)
    print("data final check:",final_data_counter)
    print("whole dataset:",whole_data_counter)


    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = dataset_statistics
    return dataset,outlier_counter1,traj_counter1,outlier_counter2,traj_counter2, outlier_dataset


def make_interleaved_dataset(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs. Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overidden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overidden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # default to uniform sampling
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = []
    for dataset_kwargs in dataset_kwargs_list:
        _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics.append(dataset_statistics)

    # balance and normalize weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # allocate threads based on weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    logging.info("Threads per dataset: %s", threads_per_dataset)
    logging.info("Reads per dataset: %s", reads_per_dataset)

    # construct datasets
    datasets = []
    for dataset_kwargs, dataset_statistics, threads, reads in zip(
        dataset_kwargs_list,
        all_dataset_statistics,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=dataset_statistics,
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        datasets.append(dataset)

    # interleave at the frame level and then shuffle
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    # apply frame transforms
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # sequential batch (parallel batch seems to use much more memory)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = all_dataset_statistics
    dataset.sample_weights = sample_weights
    return dataset
