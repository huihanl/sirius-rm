"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils
import robomimic.utils.file_utils as FileUtils

import wandb

MODEL_DEMO = 0.0
MODEL_ROLLOUT = 1.0
MODEL_INTV = 2.0
MODEL_PREINTV = 3.0

REAL_DEMO = -1.0
REAL_ROLLOUT = 0.0
REAL_INTV = 1.0
REAL_PREINTV = -10.0

model_to_real = {
    MODEL_DEMO: REAL_DEMO,
    MODEL_ROLLOUT: REAL_ROLLOUT,
    MODEL_INTV: REAL_INTV,
    MODEL_PREINTV: REAL_PREINTV,
}

real_to_model = {
    REAL_DEMO: MODEL_DEMO,
    REAL_ROLLOUT: MODEL_ROLLOUT,
    REAL_INTV: MODEL_INTV,
    REAL_PREINTV: MODEL_PREINTV,
}

import sys

sys.path.append('/home/huihanl/railrl-private/hitl_ws/classifier')
sys.path.append('/scratch/cluster/huihanl/railrl-private/hitl_ws/classifier')


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hdf5_path,
            obs_keys,
            dataset_keys,
            frame_stack=1,
            seq_length=1,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=True,
            num_eps=None,  # for loading different num of trajectories
            sort_demo_key=None,
            use_gripper_history=False,  # if using gripper history
            hc_weights_dict=None,
            use_sampler=False,
            prioritize_first_sampler=False,
            prioritize_first_weight=1.0,
            classifier_sampler=None,
            classifier_use_weighted_loss=False,
            remove_intervention_sampler=False,
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)
        Args:
            hdf5_path (str): path to hdf5
            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset
            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset
            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).
            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).
            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).
            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).
            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.
            goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals
            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all
                non-image data. Set to None to use no caching - in this case, every batch sample is
                retrieved via file i/o. You should almost never set this to None, even for large
                image datasets.
            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.
            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.
            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load
            load_next_obs (bool): whether to load next_obs from the dataset
        """
        super(SequenceDataset, self).__init__()

        self.use_sampler = use_sampler
        self.prioritize_first_sampler = prioritize_first_sampler
        self.prioritize_first_weight = prioritize_first_weight
        
        self.classifier_sampler = classifier_sampler
        self.classifier_use_weighted_loss = classifier_use_weighted_loss
        self.remove_intervention_sampler = remove_intervention_sampler

        self.hc_weights_dict = hc_weights_dict

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = False #load_next_obs
        self.filter_by_attribute = filter_by_attribute

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"  # we use last next_obs as goal

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.num_eps = num_eps
        self.sort_demo_key = sort_demo_key
        self.use_gripper_history = use_gripper_history
        self.load_demo_info(filter_by_attribute=self.filter_by_attribute,
                            num_eps=self.num_eps,
                            sort_demo_key=self.sort_demo_key,
                            )

        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # data to be overriden
        self._data_override = {demo_id: dict() for demo_id in self.demos}

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory
            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs,
                use_gripper_history=self.use_gripper_history,
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # debug for IWR: should not delete here, need hdf5 cache later
                if self.__class__.__name__ not in ["IWRDataset", "ClassifierRelabeledDataset", "WeightedDataset"]:
                    self._delete_hdf5_cache()
        else:
            self.hdf5_cache = None
            
        self._classifier_weight = None
        if self.classifier_use_weighted_loss:
            self._classifier_weight = self._get_classifier_weight()
        
        self.close_and_delete_hdf5_handle()

    def sort_demos(self, key):

        def rollout_p(d):
            action_modes = self.hdf5_file["data/" + d]["action_modes"][()]
            rollout_percentage = sum(abs(action_modes) != 1) / len(action_modes)
            return rollout_percentage

        def round_info(d):
            try:
                round_num = self.hdf5_file["data/" + d]["round"][()][0]
            except:
                round_num = 3
            if round_num == 0: # confirm round info is correct
                assert (self.hdf5_file["data/" + d]["action_modes"][()] == -1).all()
            return round_num

        print(key)
        assert key in ["MFI", "LFI", "FILO", "FIFO"]
       
        import random
        random.shuffle(self.demos)

        if key == "LFI": 
            self.demos.sort(key=lambda x: rollout_p(x), reverse=True)
        elif key == "MFI":
            self.demos.sort(key=lambda x: rollout_p(x))
        elif key == "FILO":
            self.demos.sort(key=lambda x: round_info(x))
        else:
            self.demos.sort(key=lambda x: round_info(x), reverse=True)

    def load_demo_info(self,
                       filter_by_attribute=None,
                       demos=None,
                       num_eps=None,
                       sort_demo_key=None,
                       ):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load
            demos (list): list of demonstration keys to load from the hdf5 file. If
                omitted, all demos in the file (or under the @filter_by_attribute
                filter key) are used.
        """
        # filter demo trajectory by mask
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            if type(filter_by_attribute) is str:
                self.demos = [elem.decode("utf-8") for elem in
                              np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
            elif type(filter_by_attribute) is list:
                demos_lst = []
                for filter_key in filter_by_attribute:
                    demos_lst.extend([elem.decode("utf-8") for elem in
                              np.array(self.hdf5_file["mask/{}".format(filter_key)][:])])
                assert len(demos_lst) == len(set(demos_lst))
                self.demos = demos_lst
            else:
                raise Error
        else:
            self.demos = list(self.hdf5_file["data"].keys())

        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        if sort_demo_key is not None:
            self.sort_demos(sort_demo_key)

        if num_eps is not None:
            self.demos = self.demos[:num_eps]  # choose the first num_eps

        self.n_demos = len(self.demos)

        print("number of demos: ", self.n_demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs, use_gripper_history):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.
        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
            load_next_obs (bool): whether to load next_obs from the dataset
        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            if not use_gripper_history:
                # get obs
                all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in
                                       obs_keys}
                if load_next_obs:
                    all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype('float32')
                                                for k in obs_keys}
            else:
                # get obs with gripper history information
                all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in
                                       obs_keys}
                history = self._get_gripper_history(all_data[ep]["obs"])
                all_data[ep]["obs"]["robot0_gripper_qpos"] = history
                if load_next_obs:
                    all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype('float32')
                                                for k in obs_keys}
                    history = self._get_gripper_history(all_data[ep]["next_obs"])
                    all_data[ep]["next_obs"]["robot0_gripper_qpos"] = history
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    if k == "pull_rollout":
                        if (hdf5_file["data/{}/action_modes".format(ep)][()] == 0).all():
                            all_data[ep][k] = np.array([1] * len(hdf5_file["data/{}/action_modes".format(ep)][()])).astype('float32')
                        else:
                            all_data[ep][k] = np.array([0] * len(hdf5_file["data/{}/action_modes".format(ep)][()])).astype('float32')
                        continue
                    elif k == "round":
                        all_data[ep][k] = np.array([3] * len(hdf5_file["data/{}/states".format(ep)][()])).astype('float32')
                        continue
                    raise ValueError("key {} does not exist!".format(k))
                    # all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data

    def _get_gripper_history(self, obs_data):
        gripper = obs_data["robot0_gripper_qpos"]
        gripper_start = np.array([gripper[0] for i in range(4)])
        gripper_with_history = np.concatenate([gripper_start, gripper], axis=0)
        history = np.array([np.reshape(gripper_with_history[i:i + 5], 10) for i in range(len(gripper))])
        return history

    def normalize_obs(self):
        """
        Computes a dataset-wide mean and standard deviation for the observations
        (per dimension and per obs key) and returns it.
        """

        def _compute_traj_stats(traj_obs_dict):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = {k: {} for k in traj_obs_dict}
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True)  # [1, ...]
                traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0,
                                                                                                keepdims=True)  # [1, ...]
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        merged_stats = _compute_traj_stats(obs_traj)
        print("SequenceDataset: normalizing observations...")
        for ep in LogUtils.custom_tqdm(self.demos[1:]):
            obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
            obs_traj = ObsUtils.process_obs_dict(obs_traj)
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = {k: {} for k in merged_stats}
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
            obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
        return obs_normalization_stats

    def get_obs_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.
        Returns:
            obs_normalization_stats (dict): a dictionary for observation
                normalization. This maps observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        assert self.hdf5_normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            try:
                ret = self.hdf5_file[hd5key]
            except:
                import pdb; pdb.set_trace()
                self.hdf5_file[hd5key]
                print("hd5key missed: ", hd5key)

        # override as necessary
        ret = self._data_override[ep].get(key, ret)

        return ret

    def __getitem__(self, index):
        
        meta = self.get_item(index)
        
        if self.classifier_use_weighted_loss:
            assert self._classifier_weight is not None
            meta["classifier_weights"] = self._classifier_weight[index]
        return meta 

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )
        if self.hdf5_normalize_obs:
            meta["obs"] = ObsUtils.normalize_obs(meta["obs"], obs_normalization_stats=self.obs_normalization_stats)

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )
            if self.hdf5_normalize_obs:
                meta["next_obs"] = ObsUtils.normalize_obs(meta["next_obs"],
                                                          obs_normalization_stats=self.obs_normalization_stats)

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                goal = ObsUtils.normalize_obs(goal, obs_normalization_stats=self.obs_normalization_stats)
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        meta["index"] = index

        return meta

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index: seq_end_index].astype("float32")

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1,
                                   prefix="obs"):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"
        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        # prepare image observations from dataset
        return ObsUtils.process_obs_dict(obs)

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=0,  # don't frame stack for meta keys
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self.obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta

    def _get_is_first_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=["is_first"],
            seq_length=self.seq_length
        )

        self.action_mode_selection = 0 # only selection = 0 has the first
        return meta['is_first'][self.action_mode_selection]

    def _prioritize_first_sampler(self):

        self.action_mode_selection = 0

        self.action_mode_cache = np.array([self._get_is_first_mode(i) for i in range(len(self))])

        print("Number of seqs that is_first: ", sum(self.action_mode_cache == 1))

        weights = np.zeros(len(self))

        for index in range(len(self)):
            # intervention (s, a) get up-weighted
            is_first = self.action_mode_cache[index] == 1
            if is_first:
                num_int = np.sum(self.action_mode_cache == 1)
                weights[index] = self.prioritize_first_weight
            else:
                weights[index] = 1.

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _get_failure_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=["three_class"],
            seq_length=self.seq_length
        )

        return meta['three_class'][-1]

    def get_failure_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        weights = np.zeros(len(self))

        self.failure_mode_cache = np.array([self._get_failure_mode(i) for i in range(len(self))])

        for index in range(len(self)):
            # intervention (s, a) get up-weighted
            is_intervention = self.failure_mode_cache[index] == -1
            if is_intervention:
                num_int = np.sum(self.failure_mode_cache == -1) # negative reward for failure
                weights[index] = (len(self.failure_mode_cache) - num_int) / num_int
            else:
                weights[index] = 1.

        print(weights.max(), weights.min(), weights.mean())

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def get_3class_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        weights = np.zeros(len(self))

        self.failure_mode_cache = np.array([self._get_failure_mode(i) for i in range(len(self))])

        num_int = np.sum(self.failure_mode_cache == -2)
        num_rollouts = np.sum(self.failure_mode_cache == 0)
        num_pre_intv = np.sum(self.failure_mode_cache == -1)

        for index in range(len(self)):
            is_intervention = self.failure_mode_cache[index] == -2
            is_rollouts = self.failure_mode_cache[index] == 0
            is_pre_intv = self.failure_mode_cache[index] == -1

            if is_intervention:
                weights[index] = 1
            if is_rollouts:
                weights[index] = 1 / (num_rollouts / num_int)
            if is_pre_intv:
                weights[index] = 1 / (num_pre_intv / num_int)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _get_classifier_weight(self):
        
        weights = np.zeros(len(self))

        self.failure_mode_cache = np.array([self._get_failure_mode(i) for i in range(len(self))])

        num_int = np.sum(self.failure_mode_cache == -2)
        num_rollouts = np.sum(self.failure_mode_cache == 0)
        num_pre_intv = np.sum(self.failure_mode_cache == -1)

        for index in range(len(self)):
            is_intervention = self.failure_mode_cache[index] == -2
            is_rollouts = self.failure_mode_cache[index] == 0
            is_pre_intv = self.failure_mode_cache[index] == -1

            if is_intervention:
                weights[index] = 1
            if is_rollouts:
                weights[index] = 1 / (num_rollouts / num_int)
            if is_pre_intv:
                weights[index] = 1 / (num_pre_intv / num_int)
                
        weights /= weights.mean()
        print("Classifier weights: ", weights.max(), weights.min(), weights.mean())
        return weights

    def get_action_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=["action_modes"],
            seq_length=self.seq_length
        )

        return meta['action_modes'][-1]

    def get_remove_intervention_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        action_mode_cache = np.array([self.get_action_mode(i) for i in range(len(self))])

        weights = np.zeros(len(self))

        for index in range(len(self)):
            # intervention (s, a) get up-weighted
            is_intervention = action_mode_cache[index] == 1
            if is_intervention:
                weights[index] = 0.
            else:
                weights[index] = 1.

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """

        if not self.use_sampler: 
            return None

        if self.remove_intervention_sampler:
            return self.get_remove_intervention_sampler()

        if self.classifier_sampler is not None:
            if self.classifier_sampler == "3class":
                return self.get_3class_sampler()
            elif self.classifier_sampler == "2class":
                return self.get_failure_sampler()
            else:
                raise ValueError("Invalid classifier sampler: {}".format(self.classifier_sampler))

        if self.prioritize_first_sampler:
            return self._prioritize_first_sampler()
        
        weights = np.ones(len(self))

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _delete_hdf5_cache(self):
        # don't need the previous cache anymore
        del self.hdf5_cache
        self.hdf5_cache = None


class BootstrapDataset(SequenceDataset):
    def get_dataset_sampler(self):

        subsample_idx = np.random.choice(range(len(self)), len(self), replace=True)

        weights = np.zeros(len(self))
        for i in range(len(self)):
            weights[i] = np.count_nonzero(subsample_idx == i)
        
        print("==========================")
        print("Subsamples properties")

        for i in range(int(max(weights)) + 1):
            print("appear {} times: ".format(i), np.count_nonzero(weights == i))

        assert sum(weights) == len(self) # same number of samples with original dataset

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler


class IWRDataset(SequenceDataset):
    """
    A dataset class that is useful for performing dataset operations needed by some
    human-in-the-loop algorithms, such as labeling good and bad transitions depending
    on when "interventions" occurred. This information can be used by algorithms
    that perform policy regularization.
    """

    def __init__(self, 
            action_mode_selection=0,
            *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert "action_modes" in self.dataset_keys

        self.action_mode_selection = action_mode_selection
        assert action_mode_selection in [0, -1] # only start and end

        self.action_mode_cache = np.array([self.get_action_mode(i) for i in range(len(self))])

        if self.hdf5_cache_mode == "all":
            self._delete_hdf5_cache()

    def get_action_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=["action_modes"],
            seq_length=self.seq_length
        )

        return meta['action_modes'][self.action_mode_selection]

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        weights = np.zeros(len(self))

        for index in range(len(self)):
            # intervention (s, a) get up-weighted
            is_intervention = self.action_mode_cache[index] == 1
            if is_intervention:
                num_int = np.sum(self.action_mode_cache == 1)
                weights[index] = (len(self.action_mode_cache) - num_int) / num_int
            else:
                weights[index] = 1.

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler


class WeightedDatasetClassifier(SequenceDataset):
    """
    A dataset class that is useful for performing dataset operations needed by some
    human-in-the-loop algorithms, such as labeling good and bad transitions depending
    on when "interventions" occurred. This information can be used by algorithms
    that perform policy regularization.
    """
    def __init__(self, 
            use_hc_weights=False,
            weight_key="intv_labels",
            w_demos=10,
            w_rollouts=1,
            w_intvs=10,
            w_pre_intvs=0.1,
            normalize_weights=False,
            update_weights_at_init=True,
            traj_label_type="last",
            use_weighted_sampler=False,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_hc_weights = use_hc_weights
        self.weight_key = weight_key
        self.demos = w_demos
        self.rollouts = w_rollouts
        self.intvs = w_intvs
        self.pre_intvs = w_pre_intvs
        self.normalize_weights = normalize_weights

        assert self.weight_key in self.dataset_keys

        self.weight_key = self.weight_key 

        self.traj_label_type = traj_label_type

        self.action_mode_cache = np.array([self.get_action_mode(i) for i in range(len(self))])

        if self.hdf5_cache_mode == "all":
            self._delete_hdf5_cache()

    def get_action_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=[self.weight_key],
            seq_length=self.seq_length
        )

        if self.traj_label_type == "first":
            label_idx = 0
        elif self.traj_label_type == "last":
            label_idx = -1
        elif self.traj_label_type == "middle":
            label_idx = self.seq_length // 2

        return meta[self.weight_key][label_idx]

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        weights = np.zeros(len(self))
        

        if self.use_hc_weights:
            weight_dict = {
                -1.0: self.demos,
                0.0: self.rollouts,
                1.0: self.intvs,
                -10.0: self.pre_intvs,
                
            }

            for index in range(len(self)):
                # intervention (s, a) get up-weighted
                for label in weight_dict:
                    if self.action_mode_cache[index] == label:
                        weights[index] = weight_dict[label]
                        if weight_dict[label] == -1: # default to using RATIO
                            num_int = np.sum(self.action_mode_cache == 1)
                            num_this_class = np.sum(self.action_mode_cache == label)
                            weights[index] = 1 / (num_this_class / num_int)
        else:
            for index in range(len(self)):
                is_intervention = self.action_mode_cache[index] == 1
                is_demos = self.action_mode_cache[index] == -1
                is_rollouts = self.action_mode_cache[index] == 0
                is_pre_intv = self.action_mode_cache[index] == -10

                num_int = np.sum(self.action_mode_cache == 1)
                num_demos = np.sum(self.action_mode_cache == -1)
                num_rollouts = np.sum(self.action_mode_cache == 0)
                num_pre_intv = np.sum(self.action_mode_cache == -10)

                if is_intervention:
                    weights[index] = 1
                if is_demos:
                    weights[index] = 1 / (num_demos / num_int)
                if is_rollouts:
                    weights[index] = 1 / (num_rollouts / num_int)
                if is_pre_intv:
                    weights[index] = 1 / (num_pre_intv / num_int)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler


class WeightedDataset(SequenceDataset):
    def __init__(
            self,
            use_hc_weights=False,
            weight_key="intv_labels",
            w_demos=10,
            w_rollouts=1,
            w_intvs=10,
            w_pre_intvs=0.1,
            normalize_weights=False,
            update_weights_at_init=True,
            use_weighted_sampler=False,
            use_iwr_ratio=False,
            iwr_ratio_adjusted=False,
            action_mode_selection=0,
            same_weight_for_seq=False,
            use_category_ratio=False,
            prenormalize_weights=False,
            give_final_percentage=False,
            ours_percentage=False,

            # upweight different rounds of data
            diff_weights_diff_rounds=False,
            rounds_key="round",
            round_upweights={4 : 1.2}, # round num : weight
            normalize_after_round_upweight=False,

            rounds_resampling=False,
            resampling_weight=1.,

            delete_rollout_ratio=-1,
            use_novelty=False,

            memory_org_type=None,

            pure_rollout_key="pull_rollout",

            not_use_preintv=False,

            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.use_hc_weights = use_hc_weights
        self.weight_key = weight_key
        self.w_demos = w_demos
        self.w_rollouts = w_rollouts
        self.w_intvs = w_intvs
        self.w_pre_intvs = w_pre_intvs
        self.normalize_weights = normalize_weights
        self.use_weighted_sampler = use_weighted_sampler
        self.use_iwr_ratio = use_iwr_ratio
        self.iwr_ratio_adjusted = iwr_ratio_adjusted
        self.action_mode_selection = action_mode_selection
        self.same_weight_for_seq = same_weight_for_seq

        self.use_category_ratio = use_category_ratio
        self.prenormalize_weights = prenormalize_weights
        self.give_final_percentage = give_final_percentage
        self.ours_percentage = ours_percentage

        self.diff_weights_diff_rounds = diff_weights_diff_rounds
        self.rounds_key = rounds_key
        self.round_upweights = round_upweights
        self.normalize_after_round_upweight = normalize_after_round_upweight

        self.rounds_resampling = rounds_resampling
        #assert self.rounds_resampling + self.diff_weights_diff_rounds <= 1
        self.resampling_weight = resampling_weight

        self.delete_rollout_ratio = delete_rollout_ratio
        self.use_novelty = use_novelty

        self.memory_org_type = memory_org_type

        self.pure_rollout_key = pure_rollout_key

        assert (self.__class__.__name__ == "NoveltyRelabeledDataset" and self.use_novelty) or \
               (self.__class__.__name__ != "NoveltyRelabeledDataset" and not self.use_novelty)

        assert use_category_ratio + \
               use_iwr_ratio + \
               prenormalize_weights + \
               give_final_percentage + \
               ours_percentage <= 1

        if self.delete_rollout_ratio > 0:
            assert not self.use_novelty

        assert action_mode_selection in [0, -1]

        self._weights = np.ones((len(self), self.seq_length))
        if update_weights_at_init:
            self._update_weights()

        self.not_use_preintv = not_use_preintv
        if self.not_use_preintv:
            self.action_mode_selection = -1 # for sampling purpose

    def _upweight_round(self, weights, rounds_labels):
        for round_num in self.round_upweights:
            inds = np.where(rounds_labels == int(round_num))
            if len(inds[0]) == 0:
                continue
            print(
                "reweight round {} with weight {}".format(round_num,
                                                            self.round_upweights[round_num])
                  )
            weights[inds] *= self.round_upweights[round_num]
        return weights

    def _get_rounds_labels(self):
        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.rounds_key,),
                seq_length=self.seq_length,
            )[self.rounds_key]
            labels.append(label)

        labels = np.stack(labels)
        return labels

    def _get_action_mode(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=[self.weight_key],
            seq_length=self.seq_length
        )
        return meta[self.weight_key][self.action_mode_selection]

    def _get_iwr_ratio(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])
        
        num_int = np.sum(self.action_mode_cache == 1)
        total_num = len(self.action_mode_cache)
        weight_intv = (total_num - num_int) / num_int
        self.w_demos = 1.
        self.w_rollouts = 1.
        self.w_intvs = weight_intv
        self.w_pre_intvs = 1.

        if self.iwr_ratio_adjusted:
            if num_int != 0:
                weight_intv = total_num / (2 * num_int) 
                weight_non_intv = total_num / (2 * (total_num - num_int)) 
            else:
                weight_intv = 1
                weight_non_intv = 1
            self.w_demos = weight_non_intv
            self.w_rollouts = weight_non_intv
            self.w_intvs = weight_intv
            self.w_pre_intvs = weight_non_intv

    def _get_category_ratio(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        total_num = len(self.action_mode_cache)
        weight_intv = total_num / num_int
        weight_demos = total_num / num_demos

        self.w_demos = weight_demos
        self.w_intvs = weight_intv
        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)

    def _prenormalize_weights(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        total_num = len(self.action_mode_cache)
        weight_intv = total_num / num_int
        weight_demos = total_num / num_demos
        weight_rollouts = total_num / num_rollouts
        weight_pre_intv = total_num / num_pre_intv

        self.w_demos *= weight_demos
        self.w_intvs *= weight_intv
        self.w_rollouts *= weight_rollouts
        self.w_pre_intvs *= weight_pre_intv

        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)
        print("rollouts weight: ", self.w_rollouts)
        print("preintv weight: ", self.w_pre_intvs)

    def _weight_from_percentage(self):

        if self.not_use_preintv:
            assert self.action_mode_selection == -1
        
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        total_num = len(self.action_mode_cache)
        ratio_intv = num_int / total_num
        ratio_demos = num_demos / total_num
        ratio_rollouts = num_rollouts / total_num
        ratio_pre_intv = num_pre_intv / total_num

        self.w_demos /= ratio_demos
        self.w_intvs /= ratio_intv
        self.w_rollouts /= ratio_rollouts
        self.w_pre_intvs /= ratio_pre_intv

        """
        weight_intv = self.w_intvs
        weight_preintv = self.w_pre_intvs

        self.w_demos = 1
        self.w_intvs = weight_intv / ratio_intv
        self.w_rollouts = (1 - weight_intv - ratio_demos - weight_preintv) / ratio_rollouts
        self.w_pre_intvs /= ratio_pre_intv
        """
 
        if self.not_use_preintv:
           self.w_pre_intvs = 0

        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)
        print("rollouts weight: ", self.w_rollouts)
        print("preintv weight: ", self.w_pre_intvs)

    def _weight_from_percentage_delete_rollouts(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        delete_rollout_num = num_rollouts * self.delete_rollout_ratio
        num_rollouts = num_rollouts - delete_rollout_num

        total_num = len(self.action_mode_cache) - delete_rollout_num
        ratio_intv = num_int / total_num
        ratio_demos = num_demos / total_num
        ratio_rollouts = num_rollouts / total_num
        ratio_pre_intv = num_pre_intv / total_num

        if self.w_rollouts == 0:
            ratio_rollouts = 1

        self.w_demos /= ratio_demos
        self.w_intvs /= ratio_intv
        self.w_rollouts /= ratio_rollouts
        self.w_pre_intvs /= ratio_pre_intv
        
        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)
        print("rollouts weight: ", self.w_rollouts)
        print("preintv weight: ", self.w_pre_intvs)

    def _ours_percentage(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        total_num = len(self.action_mode_cache)
        ratio_intv = num_int / total_num
        ratio_demos = num_demos / total_num
        ratio_rollouts = num_rollouts / total_num
        ratio_pre_intv = num_pre_intv / total_num

        weight_intv = 0.5
        weight_preintv = 0.002

        self.w_demos = 1
        self.w_intvs = weight_intv / ratio_intv
        self.w_rollouts = (1 - weight_intv - ratio_demos - weight_preintv) / ratio_rollouts
        self.w_pre_intvs = weight_preintv / ratio_pre_intv

        print("ratio_intv: ", ratio_intv)
        print("ratio_pre_intv: ", ratio_pre_intv)

        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)
        print("rollouts weight: ", self.w_rollouts)
        print("preintv weight: ", self.w_pre_intvs)

    def _ours_percentage_delete_rollouts(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        delete_rollout_num = int(num_rollouts * self.delete_rollout_ratio)
        num_rollouts = num_rollouts - delete_rollout_num

        total_num = len(self.action_mode_cache) - delete_rollout_num

        ratio_intv = num_int / total_num
        ratio_demos = num_demos / total_num
        ratio_rollouts = num_rollouts / total_num
        ratio_pre_intv = num_pre_intv / total_num

        weight_intv = 0.5
        weight_preintv = 0.002

        self.w_demos = 1
        self.w_intvs = weight_intv / ratio_intv
        self.w_rollouts = (1 - weight_intv - ratio_demos - weight_preintv) / ratio_rollouts
        self.w_pre_intvs = weight_preintv / ratio_pre_intv

        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)
        print("rollouts weight: ", self.w_rollouts)
        print("preintv weight: ", self.w_pre_intvs)

    def _ours_percentage_delete_rollouts_from_novelty(self):

        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        num_int = np.sum(self.action_mode_cache == 1)
        num_demos = np.sum(self.action_mode_cache == -1)
        num_rollouts = np.sum(self.action_mode_cache == 0)
        num_pre_intv = np.sum(self.action_mode_cache == -10)

        self.novelty_mode_cache = np.array([self._get_novelty_mode(i) for i in range(len(self))])

        novelty_deleted = np.sum(self.novelty_mode_cache == 0)
        num_rollouts = num_rollouts - novelty_deleted

        total_num = len(self.action_mode_cache) - novelty_deleted

        ratio_intv = num_int / total_num
        ratio_demos = num_demos / total_num
        ratio_rollouts = num_rollouts / total_num
        ratio_pre_intv = num_pre_intv / total_num

        weight_intv = 0.5
        weight_preintv = 0.002

        self.w_demos = 1
        self.w_intvs = weight_intv / ratio_intv
        self.w_rollouts = (1 - weight_intv - ratio_demos - weight_preintv) / ratio_rollouts
        self.w_pre_intvs = weight_preintv / ratio_pre_intv

        print("demos weight: ", self.w_demos)
        print("intv weight: ", self.w_intvs)
        print("rollouts weight: ", self.w_rollouts)
        print("preintv weight: ", self.w_pre_intvs)

    def _get_novelty_mode(self, index):

        novelty_key = "novelty"

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=[novelty_key],
            seq_length=self.seq_length
        )
        return meta[novelty_key][self.action_mode_selection]

    def _update_weights(self):
        print("Updating weights...")
        if not self.use_hc_weights or self.use_weighted_sampler:
            self._weights = np.ones(len(self))
            print("Done.")
            return

        if self.memory_org_type is not None:
            self._get_deleted_index_memory(self.memory_org_type)
            assert self.memory_save_cache is not None

        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.weight_key,),
                seq_length=self.seq_length,
            )[self.weight_key]

            if self.same_weight_for_seq:
                label = np.array([label[self.action_mode_selection]] * self.seq_length)

            labels.append(label)

        labels = np.stack(labels)

        if self.use_iwr_ratio:
            self._get_iwr_ratio()

        if self.use_category_ratio:
            self._get_category_ratio()

        if self.prenormalize_weights:
            self._prenormalize_weights()

        if self.give_final_percentage:
            if self.delete_rollout_ratio <= 0:
                self._weight_from_percentage()
            else:
                self._weight_from_percentage_delete_rollouts()

        if self.ours_percentage:
            if self.use_novelty:
                self._ours_percentage_delete_rollouts_from_novelty()
            elif self.delete_rollout_ratio > 0:
                self._ours_percentage_delete_rollouts()
            else:
                self._ours_percentage()

        weight_dict = {
            -1: self.w_demos,
            0: self.w_rollouts,
            1: self.w_intvs,
            -10: self.w_pre_intvs,
        }

        weights = np.ones((len(self), self.seq_length))
        assert weights.shape == labels.shape

        for (l, w) in weight_dict.items():
            inds = np.where(labels == l)
            weights[inds] = w

        print("weights mean: ", np.mean(weights))

        if self.normalize_weights or self.not_use_preintv: # normalize since preintv is gone
            print("Mean weight before normalization", np.mean(weights))
            weights /= np.mean(weights)
            print("Mean weight after normalization", np.mean(weights))

        if self.diff_weights_diff_rounds:
            """ Upweight certain round """
            
            print()
            print("*" * 50)
            print()

            round_labels = self._get_rounds_labels()
            weights_prev = weights.copy()
            weights = self._upweight_round(weights, round_labels)
            assert not (weights_prev == weights).all()
            print("Mean weight before normalization, last round upweighted", np.mean(weights))
            if self.normalize_after_round_upweight:
                weights /= np.mean(weights)
                print("Mean weight after normalization", np.mean(weights))

            print()

        self._weights = weights
        #self._weights = self._weights.round(decimals=4)
        print("Done.")

    def _get_round_label_per_sample(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=[self.rounds_key],
            seq_length=self.seq_length
        )
        return meta[self.rounds_key][-1]

    def _get_pure_label_per_sample(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=[self.pure_rollout_key],
            seq_length=self.seq_length
        )
        return meta[self.pure_rollout_key][-1]

    def _get_deleted_index_memory(self, org_type):
        assert org_type in ["FIFO", "FILO", "LFI", "MFI", "Random"]
        if org_type == "Random":
            self._get_deleted_index_memory_random()
        elif org_type in ["FIFO", "FILO"]:
            self._get_deleted_index_memory_SEQ(org_type)
        else:
            self._get_deleted_index_memory_INTV(org_type)

    def _get_deleted_index_memory_random(self):
        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])
        
        ones = np.ones(self.action_mode_cache.shape)
        zeros = np.zeros(self.action_mode_cache.shape)

        self.memory_save_cache = np.where(self.action_mode_cache == 0, zeros, ones)
        self.memory_save_cache = np.where(self.action_mode_cache == -10, zeros, self.memory_save_cache)

        num_rollouts = np.sum(self.action_mode_cache == 0) + np.sum(self.action_mode_cache == -10)
        delete_num = self.delete_rollout_ratio * num_rollouts
        saved_num = int((1 - self.delete_rollout_ratio) * num_rollouts)

        inds_rollout = np.where(self.action_mode_cache == 0)[0]
        inds_preintv = np.where(self.action_mode_cache == -10)[0]
        print("original preintv: ", len(inds_preintv))
        inds_choice = np.union1d(inds_rollout, inds_preintv)

        saved_idx = np.random.choice(inds_choice, saved_num, replace=False)
        self.memory_save_cache[saved_idx] = 1.

        print("cache sum: ", sum(self.memory_save_cache))
        # print("non rollout: ", sum(self.action_mode_cache != 0))
        # print("saved rollout: ", int((1 - self.delete_rollout_ratio) * num_rollouts))
 
        print("saved 1: ", sum(self.memory_save_cache[self.action_mode_cache == 1]))
        print("saved -1: ", sum(self.memory_save_cache[self.action_mode_cache == -1]))
        print("saved -10: ", sum(self.memory_save_cache[self.action_mode_cache == -10]))
        print("saved 0: ", sum(self.memory_save_cache[self.action_mode_cache == 0]))
        print("other: ", int((1 - self.delete_rollout_ratio) * num_rollouts))

        assert sum(self.memory_save_cache) == sum(self.action_mode_cache == 1) + \
               sum(self.action_mode_cache == -1) + \
               int((1 - self.delete_rollout_ratio) * num_rollouts)
        print("preserve ratio: ", np.sum(self.memory_save_cache) / self.memory_save_cache.shape[0])

        print()
        print("====== RATIO ======")
        print("rollout: ", sum(self.action_mode_cache == 0) / len(self.action_mode_cache))
        print("intv: ", sum(self.action_mode_cache == 1) / len(self.action_mode_cache))
        print("demos: ", sum(self.action_mode_cache == -1) / len(self.action_mode_cache))
        print("pre-intv: ", sum(self.action_mode_cache == -10) / len(self.action_mode_cache))

    def _get_deleted_index_memory_INTV(self, org_type):

        PURE_ROLLOUT = 1
        INTVED_ROLLOUT = 0

        type_dict = {0: "INTVED_ROLLOUT",
                     1: "PURE_ROLLOUT"}

        self.pure_label_cache = np.array([self._get_pure_label_per_sample(i) for i in range(len(self))])

        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        ones = np.ones(self.action_mode_cache.shape)
        zeros = np.zeros(self.action_mode_cache.shape)

        self.memory_save_cache = np.where(self.action_mode_cache == 0, zeros, ones)
        self.memory_save_cache = np.where(self.action_mode_cache == -10, zeros, self.memory_save_cache)

        print(sum(self.memory_save_cache))
        print(sum(self.action_mode_cache == 1))
        print(sum(self.action_mode_cache == -1))
        assert sum(self.memory_save_cache) == sum(self.action_mode_cache == 1) + sum(self.action_mode_cache == -1)

        num_rollouts = np.sum(self.action_mode_cache == 0) + np.sum(self.action_mode_cache == -10)

        delete_num = self.delete_rollout_ratio * num_rollouts
        saved_num = int((1 - self.delete_rollout_ratio) * num_rollouts)
        num_rounds = 4
        if org_type == "LFI":
            loop_seq = [PURE_ROLLOUT, INTVED_ROLLOUT]
        elif org_type == "MFI":
            loop_seq = [INTVED_ROLLOUT, PURE_ROLLOUT]
        else:
            assert Error

        print(org_type)
        print(loop_seq)

        for rollout_type in loop_seq:
            print(rollout_type, type_dict[rollout_type])

            # round info
            #inds_round = np.where(self.rounds_label_cache == round)[0]
            inds_round = np.where(self.pure_label_cache == rollout_type)[0]

            # intv label
            inds_rollout = np.where(self.action_mode_cache == 0)[0]
            inds_preintv = np.where(self.action_mode_cache == -10)[0]
            inds_choice = np.union1d(inds_rollout, inds_preintv)

            type_rollout_idx = np.intersect1d(inds_round, inds_choice)

            this_round_labels = self.pure_label_cache[type_rollout_idx]
            indices = type_rollout_idx
            this_round_count = len(type_rollout_idx)
            rest = saved_num - this_round_count
            if rest < 0:
                saved_idx = np.random.choice(indices, saved_num, replace=False)
                self.memory_save_cache[saved_idx] = 1.
                print("Sampling {} samples from type {}".format(saved_num, type_dict[rollout_type]))
                break
            else:
                self.memory_save_cache[type_rollout_idx] = 1.
                saved_num = rest
                print("After type {}, need {} more".format(type_dict[rollout_type], rest))
        assert rest < 0

        print("cache sum: ", sum(self.memory_save_cache))
        # print("non rollout: ", sum(self.action_mode_cache != 0))
        # print("saved rollout: ", int((1 - self.delete_rollout_ratio) * num_rollouts))

        print("saved 1: ", sum(self.memory_save_cache[self.action_mode_cache == 1]))
        print("saved -1: ", sum(self.memory_save_cache[self.action_mode_cache == -1]))
        print("saved -10: ", sum(self.memory_save_cache[self.action_mode_cache == -10]))
        print("saved 0: ", sum(self.memory_save_cache[self.action_mode_cache == 0]))
        print("other: ", int((1 - self.delete_rollout_ratio) * num_rollouts))

        assert sum(self.memory_save_cache) == sum(self.action_mode_cache == 1) + \
               sum(self.action_mode_cache == -1) + \
               int((1 - self.delete_rollout_ratio) * num_rollouts)
        print("preserve ratio: ", np.sum(self.memory_save_cache) / self.memory_save_cache.shape[0])

    def _get_deleted_index_memory_SEQ(self, org_type):

        self.rounds_label_cache = np.array([self._get_round_label_per_sample(i) for i in range(len(self))])

        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        ones = np.ones(self.action_mode_cache.shape)
        zeros = np.zeros(self.action_mode_cache.shape)

        self.memory_save_cache = np.where(self.action_mode_cache == 0, zeros, ones)
        self.memory_save_cache = np.where(self.action_mode_cache == -10, zeros, self.memory_save_cache)

        print(sum(self.memory_save_cache))
        print(sum(self.action_mode_cache == 1))
        print(sum(self.action_mode_cache == -1))
        assert sum(self.memory_save_cache) == sum(self.action_mode_cache == 1) + sum(self.action_mode_cache == -1)

        num_rollouts = np.sum(self.action_mode_cache == 0) + np.sum(self.action_mode_cache == -10)

        delete_num = self.delete_rollout_ratio * num_rollouts
        saved_num = int((1 - self.delete_rollout_ratio) * num_rollouts)
        num_rounds = 4
        if org_type == "FILO":
            loop_seq = range(1, num_rounds)
        elif org_type == "FIFO":
            loop_seq = range(num_rounds - 1, 0, -1)
        else:
            assert Error

        assert len(loop_seq) == 3
        for round in loop_seq:
            print("At round {}: ", round)
            # round info
            inds_round = np.where(self.rounds_label_cache == round)[0]

            # intv label
            inds_rollout = np.where(self.action_mode_cache == 0)[0]
            inds_preintv = np.where(self.action_mode_cache == -10)[0]
            inds_choice = np.union1d(inds_rollout, inds_preintv)

            this_round_rollout_idx = np.intersect1d(inds_round, inds_choice)

            this_round_labels = self.rounds_label_cache[this_round_rollout_idx]
            indices = this_round_rollout_idx
            this_round_count = len(this_round_rollout_idx)
            rest = saved_num - this_round_count
            if rest < 0:
                saved_idx = np.random.choice(indices, saved_num, replace=False)
                self.memory_save_cache[saved_idx] = 1.
                break
            else:
                self.memory_save_cache[this_round_rollout_idx] = 1.
                saved_num = rest
                print("After round {}, need {} more".format(round, rest))

        print("cache sum: ", sum(self.memory_save_cache))
        # print("non rollout: ", sum(self.action_mode_cache != 0))
        # print("saved rollout: ", int((1 - self.delete_rollout_ratio) * num_rollouts))

        print("saved 1: ", sum(self.memory_save_cache[self.action_mode_cache == 1]))
        print("saved -1: ", sum(self.memory_save_cache[self.action_mode_cache == -1]))
        print("saved -10: ", sum(self.memory_save_cache[self.action_mode_cache == -10]))
        print("saved 0: ", sum(self.memory_save_cache[self.action_mode_cache == 0]))
        print("other: ", int((1 - self.delete_rollout_ratio) * num_rollouts))

        assert sum(self.memory_save_cache) == sum(self.action_mode_cache == 1) + \
               sum(self.action_mode_cache == -1) + \
               int((1 - self.delete_rollout_ratio) * num_rollouts)
        print("preserve ratio: ", np.sum(self.memory_save_cache) / self.memory_save_cache.shape[0])

        print()
        print("====== RATIO ======")
        print("rollout: ", sum(self.action_mode_cache == 0) / len(self.action_mode_cache))
        print("intv: ", sum(self.action_mode_cache == 1) / len(self.action_mode_cache))
        print("demos: ", sum(self.action_mode_cache == -1) / len(self.action_mode_cache))
        print("pre-intv: ", sum(self.action_mode_cache == -10) / len(self.action_mode_cache))

    def __getitem__(self, index):
        meta = super().__getitem__(index)

        if not self.use_weighted_sampler:
            meta["hc_weights"] = self._weights[index]

        return meta

    def _get_rounds_sampler(self):
        print("Creating rounds sampler...")
        weights = np.ones(len(self))
        weight_labels = self._get_weights_labels()
        round_labels = self._get_rounds_labels()

        # TODO: Only upsample intv from last round for now

        unique, counts = np.unique(round_labels, return_counts=True)
        rounds_numbers = [int(u) for u in unique]
        last_round = max(rounds_numbers)
        print("last round: ", last_round)

        inds_weights = np.where(weight_labels == 1)[0]
        inds_rounds = np.where(round_labels == last_round)

        intvs_last_round_inds = np.intersect1d(inds_weights, inds_rounds)

        weights[intvs_last_round_inds] *= self.resampling_weight
        print("max and min of weights: ", weights.max(), weights.min())

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        print("Done.")
        return sampler

    def _get_weights_labels(self):
        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.weight_key,),
                seq_length=1,
            )[self.weight_key][0]
            labels.append(label)
        labels = np.array(labels)
        return labels

    def _get_delete_rollout_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        import random
        weights = np.ones(len(self))

        num_rollout = np.sum(self.action_mode_cache == 0)

        for index in range(len(self)):
            # intervention (s, a) get up-weighted
            is_rollout = self.action_mode_cache[index] == 0
            if is_rollout:
                r = random.uniform(0, 1)
                if r < self.delete_rollout_ratio:
                    weights[index] = 0.

        print("ratio of rollout deleted: ", (len(self) - sum(weights)) / len(self))

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _get_delete_rollout_sampler_from_novelty(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        import random
        weights = np.ones(len(self))

        for index in range(len(self)):
            # not novel samples deleted
            not_novel_deleted = self.novelty_mode_cache[index] == 0
            if not_novel_deleted:
                weights[index] = 0.

        print("ratio of rollout deleted: ", (len(self) - sum(weights)) / len(self))

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _memory_org_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        print(self.memory_save_cache)
        unique, counts = np.unique(self.memory_save_cache, return_counts=True)
        print("unique value: ", unique)
        print("counts: ", counts)

        #assert list(set(self.memory_save_cache)) == [0, 1]

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.memory_save_cache,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def _no_preintv_sampler(self):

        self.action_mode_selection = -1

        self.action_mode_cache = np.array([self._get_action_mode(i) for i in range(len(self))])

        ones = np.ones(self.action_mode_cache.shape)
        zeros = np.zeros(self.action_mode_cache.shape)

        self.no_preintv_sampling = np.where(self.action_mode_cache == -10, zeros, ones)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=self.no_preintv_sampling,
            num_samples=len(self),
            replacement=True,
        )
        return sampler

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        """
        if self.not_use_preintv:
            return self._no_preintv_sampler()

        if self.memory_org_type is not None:
            return self._memory_org_sampler()

        if self.use_novelty:
            return self._get_delete_rollout_sampler_from_novelty()

        if self.delete_rollout_ratio > 0:
            return self._get_delete_rollout_sampler()

        if self.use_sampler and not self.use_weighted_sampler: # simply sample from sampler

            weights = np.ones(len(self))

            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(self),
                replacement=True,
            )
            return sampler

        if (not self.use_weighted_sampler) and (not self.rounds_resampling):
            return None

        if self.rounds_resampling:
            return self._get_rounds_sampler()

        print("Creating weighted sampler...")

        weights = np.zeros(len(self))

        labels = []
        for index in LogUtils.custom_tqdm(range(len(self))):
            demo_id = self._index_to_demo_id[index]
            demo_start_index = self._demo_id_to_start_indices[demo_id]

            # start at offset index if not padding for frame stacking
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            index_in_demo = index - demo_start_index + demo_index_offset

            label = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=(self.weight_key,),
                seq_length=1,
            )[self.weight_key][0]
            labels.append(label)
        labels = np.array(labels)

        weight_dict = {
            -1: self.w_demos,
            0: self.w_rollouts,
            1: self.w_intvs,
            -10: self.w_pre_intvs,
        }

        print()
        print("==================")
        print(weight_dict)
        print("==================")
        print()

        for (l, w) in weight_dict.items():
            inds = np.where(labels == l)[0]
            if len(inds) > 0:
                weights[inds] = w / len(inds)

        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(self),
            replacement=True,
        )
        print("Done.")
        return sampler


class PreintvRelabeledDataset(WeightedDataset):
    def __init__(
            self,
            mode='fixed',
            fixed_preintv_length=15,
            model_ckpt=None,
            model_th=-0.30,
            model_eval_mode='V',
            base_key="action_modes",
            *args, **kwargs
    ):
        super().__init__(*args, update_weights_at_init=False, **kwargs)

        self._label_key = "intv_labels"
        self._base_key = base_key

        # DO NOT REMOVE THIS LINE. because we are relabeling, check we are not caching __getitem__ calls
        assert self.hdf5_cache_mode is not "all"

        assert mode in ['fixed', 'model']
        self._mode = mode

        assert isinstance(fixed_preintv_length, int)
        self._fixed_preintv_length = fixed_preintv_length

        if self._mode == 'model':
            assert model_ckpt is not None
            model, _ = FileUtils.algo_from_checkpoint(ckpt_path=model_ckpt)
            self._model = model
        else:
            self._model = None

        assert model_eval_mode in ['V', 'Q', 'A']
        self._model_eval_mode = model_eval_mode

        self._model_th = model_th

        self._relabeled_values_cache = dict()

        print("Relabeling pre-interventions in dataset...")
        for demo_id in LogUtils.custom_tqdm(self.demos):
            demo_length = self._demo_id_to_demo_length[demo_id]
            ep_info = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=[self._base_key, "actions"],
                seq_length=demo_length
            )

            if self._model is not None:
                ep_info["obs"] = self.get_obs_sequence_from_demo(
                    demo_id,
                    index_in_demo=0,
                    keys=self._model.obs_key_shapes.keys(),
                    seq_length=demo_length
                )

            intv_labels = self._get_intv_labels(ep_info)
            self._data_override[demo_id][self._label_key] = intv_labels
        print("Done.")
        self._update_weights()  # update the weights after relabeling data

    def _get_intv_labels(self, ep_info):
        action_modes = ep_info[self._base_key]
        intv_labels = deepcopy(action_modes)

        intv_inds = np.reshape(np.argwhere(action_modes == 1), -1)
        intv_start_inds = [i for i in intv_inds if i > 0 and action_modes[i - 1] != 1]
        for i_start in intv_start_inds:
            for j in range(i_start - 1, -1, -1):
                if self._mode == 'fixed':
                    if j in intv_inds or i_start - j > self._fixed_preintv_length:
                        break
                elif self._mode == 'model':
                    ob = {k: ep_info["obs"][k][j] for k in ep_info["obs"].keys()}
                    ob = self._prepare_tensor(ob, device=self._model.device)

                    if self._model_eval_mode == 'V':
                        val = self._model.get_v_value(obs_dict=ob)
                    elif self._model_eval_mode == 'Q':
                        raise NotImplementedError
                    elif self._model_eval_mode == 'A':
                        raise NotImplementedError
                    else:
                        raise ValueError

                    if j in intv_inds or val > self._model_th:
                        break
                else:
                    raise ValueError

                intv_labels[j] = -10
        return intv_labels

    def _prepare_tensor(self, tensor, device=None):
        """
        Prepare raw observation dict from environment for policy.
        Args:
            ob (dict): single observation dictionary from environment (no batch dimension,
                and np.array values for each key)
        """
        tensor = TensorUtils.to_tensor(tensor)
        tensor = TensorUtils.to_batch(tensor)
        if device is not None:
            tensor = TensorUtils.to_device(tensor, device)
        tensor = TensorUtils.to_float(tensor)
        return tensor


class NoveltyRelabeledDataset(PreintvRelabeledDataset):
    def __init__(
            self,
            model_ensemble_ckpts=[],
            novelty_region=15,
            relabel_strategy="only_succ_rollout",
            *args, **kwargs
    ):
        super().__init__(
                         use_novelty=True,
                         *args,
                         **kwargs)

        self._novelty_key = "novelty"
        self._ensemble = []
        self._novelty_region = novelty_region
        self._relabel_strategy = relabel_strategy
        assert self._relabel_strategy in ["only_succ_rollout", "all"]

        # DO NOT REMOVE THIS LINE. because we are relabeling, check we are not caching __getitem__ calls
        assert self.hdf5_cache_mode is not "all"
        assert len(model_ensemble_ckpts) > 0
        for model_ckpt in model_ensemble_ckpts:
            model = FileUtils.policy_from_checkpoint(ckpt_path=model_ckpt)[0]
            self._ensemble.append(model)

        print("Relabeling novelty region in dataset...")
        for demo_id in LogUtils.custom_tqdm(self.demos):
            demo_length = self._demo_id_to_demo_length[demo_id]
            ep_info = self.get_dataset_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=[self._label_key, ],
                seq_length=demo_length
            )

            ep_info["obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self._ensemble[0].policy.obs_key_shapes.keys(),
                seq_length=demo_length
            )

            # only relabel successful rollouts
            if self._relabel_strategy == "only_succ_rollout":
                if (ep_info[self._label_key] == 0).all():
                    novel_labels = self._get_novel_labels(ep_info)
                else:
                    novel_labels = np.ones(ep_info[self._label_key].shape)
            else:
                novel_labels = self._get_novel_labels(ep_info)
            print(novel_labels)

            self._data_override[demo_id][self._novelty_key] = novel_labels
        print("Done.")
        self._update_weights()  # update the weights after relabeling data

    def _get_obs_at_idx(self, obs, i):
        d = dict()
        for key in obs:
            d[key] = obs[key][i]
        d = self._prepare_tensor(d, device=self._ensemble[0].policy.device)
        for key in d:
            d[key] = np.squeeze(d[key], 0)
        return d

    def _variance(self, obs):
        actions = []
        for policy in self._ensemble:
            action = policy(obs)
            actions.append(action)
        return np.square(np.std(np.array(actions), axis=0)).mean()

    def _get_novelty_values(self, ep_info):
        for model in self._ensemble:
            model.start_episode()
        obs = deepcopy(ep_info["obs"])
        var_lst = []
        for i in range(len(ep_info[self._label_key])):
            o = self._get_obs_at_idx(obs, i)
            var = self._variance(o)
            var_lst.append(var)
        var_lst = np.array(var_lst)
        print(var_lst.max(), var_lst.min(), var_lst.mean())
        var_lst = (var_lst > 0.07).astype(int)
        var_lst[:20] = 0 # hack for ood value at beginning
        print(var_lst)
        return var_lst

    def _get_novel_labels(self, ep_info):
        assert self._label_key == "intv_labels"
        intv_modes = ep_info[self._label_key]

        # Get high novelty values
        novelty_values = self._get_novelty_values(ep_info)
        novelty_start_inds = list(np.where(novelty_values == 1)[0])

        # Actual novelty relabeled lst
        # Default: Rollout - zero, non-rollout - one
        novelty_relabel = np.where(intv_modes != 0,
                                   np.ones(intv_modes.shape),
                                   np.zeros(intv_modes.shape)
                                   )

        # Only to relabel rollouts
        rollout_inds = np.reshape(np.argwhere(intv_modes == 0), -1)

        for i_start in novelty_start_inds:
            for j in range(i_start - 1, -1, -1):
                if i_start - j > self._novelty_region:
                    break
                if j not in rollout_inds:
                    assert novelty_relabel[j] == 1
                    continue
                novelty_relabel[j] = 1
            for j in range(i_start, len(novelty_relabel)):
                if j - i_start > self._novelty_region:
                    break
                if j not in rollout_inds:
                    assert novelty_relabel[j] == 1
                    continue
                novelty_relabel[j] = 1

        return novelty_relabel


class ClassifierRelabeledDatasetTooNew(PreintvRelabeledDataset):
    def __init__(
            self,
            classifier_ckpt,
            relabeling_filter_key=None,
            label_key="intv_labels",
            label_rollout_only=True,
            relabel_strategy="intv",  # choose from "intv", "intv_and_pre_intv", "all"
            traj_label_type="middle",
            cls_threshold=0,
            debug=False,

            pseudo_labling=False,
            demos_drop_th=0.,
            intv_drop_th=0.,
            preintv_drop_th=0.,

            rollout_demos_drop=0.,
            rollout_intvs_drop=0.,
            rollout_preintvs_drop=0.,

            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        # DO NOT REMOVE THIS LINE. because we are relabeling, check we are not caching __getitem__ calls
        assert self.hdf5_cache_mode is not "all"

        # set relabeling key
        if relabeling_filter_key is None:
            relabeling_filter_key = self.filter_by_attribute

        # determine which demos should be relabled by classifier
        if relabeling_filter_key is not None:
            self.demos_relabeling = [elem.decode("utf-8") for elem in
                                     np.array(self.hdf5_file["mask/{}".format(relabeling_filter_key)][:])]
        else:
            self.demos_relabeling = list(self.hdf5_file["data"].keys())

        # for now let's assume no frame stacking for simplicity
        assert self.n_frame_stack == 1

        assert label_key in ["action_modes", "intv_labels"]
        self._label_key = label_key
        self._label_rollout_only = label_rollout_only
        self._debug = debug
        self._cls_threshold = cls_threshold

        self._pseudo_labling = pseudo_labling
        self._demos_drop_th = demos_drop_th
        self._intv_drop_th = intv_drop_th
        self._preintv_drop_th = preintv_drop_th
        self._percentile_dict = {
            MODEL_DEMO: self._demos_drop_th,
            MODEL_INTV: self._intv_drop_th,
            MODEL_PREINTV: self._preintv_drop_th,
            MODEL_ROLLOUT: 0,
        }
        self._rollout_drop_dict = {
            MODEL_DEMO: rollout_demos_drop,
            MODEL_INTV: rollout_intvs_drop,
            MODEL_PREINTV: rollout_preintvs_drop,
            MODEL_ROLLOUT: 0,
        }

        if debug:
            self._classifier = None
            self._seq_len = 10
        else:
            self._classifier = torch.load(classifier_ckpt)
            self._seq_len = self._classifier.lstm_seq_length

        # cache of relabeled values that grows on demand
        self._relabeled_values_cache = dict()

        assert relabel_strategy in ["intv", "intv_pre_intv", "all"]
        self._relabel_strategy = relabel_strategy
        self._traj_label_type = traj_label_type

        print("Relabeling with classifier...")

        self.relabel_cache = {}
        self.logits_labels_data = []
        self.preds_labels_data = []
        self.logits_all_data = []
        self.true_labels_data = []

        #for ep in LogUtils.custom_tqdm(self.demos_relabeling):
        #    self._cache_relabeling_info_for_ep(ep)

        #self._calculate_relabeling_states()

        for ep in LogUtils.custom_tqdm(self.demos_relabeling):
            self._data_override[ep][self._label_key] = np.array(self.get_relabled_values_for_ep(ep))
        print("Done.")
        self._update_weights()

    def _calculate_relabeling_states(self):
        self.logits_labels_data = np.array(self.logits_labels_data)
        self.preds_labels_data = np.array(self.preds_labels_data)
        self.logits_all_data = np.concatenate(self.logits_all_data, 0)
        self.true_labels_data = np.array(self.true_labels_data)
        np.save("weight_data_0614", 
                {"logits_label": self.logits_labels_data, 
                 "preds_label": self.preds_labels_data, 
                 "logits_all": self.logits_all_data, 
                 "true_label": self.true_labels_data})
        """
        data = np.load("weight_data_0614.npy", allow_pickle=True)
        data_item = data.item()
        self.logits_labels_data = data_item["logits_label"]
        self.preds_labels_data = data_item["preds_label"]
        self.logits_all_data = data_item["logits_all"]
        self.true_labels_data = data_item["true_label"]
        """ 
        self.threshold_dict = {}
        self.threshold_dict[MODEL_ROLLOUT] = 0
        for cls in [MODEL_INTV, MODEL_PREINTV, MODEL_DEMO]: # 3 Classes 
            cls = int(cls)
            #print(self.logits_all_data.shape)
            this_class_all_logits = self.logits_all_data[:, cls]
            #idxs = np.argwhere(self.true_labels_data == cls)
            idxs = []
            for i in range(len(self.true_labels_data)):
                if self.true_labels_data[i] != MODEL_ROLLOUT and self.preds_labels_data[i] == cls:
                    idxs.append(i)
             
            this_class_own_logits = this_class_all_logits[idxs]
            self.threshold_dict[cls] = np.percentile(this_class_own_logits,
                                                self._percentile_dict[cls] * 100.)
            wandb.log({"{}/sample labeled to".format(cls): len(idxs),
                       "{}/samples discarded".format(cls): np.sum(this_class_own_logits < self.threshold_dict[cls]),
                       "{}/threshold".format(cls): self.threshold_dict[cls]
                       })

        self.rollouts_threshold_dict = {}
        self.rollouts_threshold_dict[MODEL_ROLLOUT] = 0
        for cls in [MODEL_INTV, MODEL_PREINTV, MODEL_DEMO]:
            cls = int(cls)
            this_class_all_logits = self.logits_all_data[:, cls]
            idxs = []
            for i in range(len(self.true_labels_data)):
                if self.true_labels_data[i] == MODEL_ROLLOUT and self.preds_labels_data[i] == cls:
                    idxs.append(i)
            this_class_robot_logits = this_class_all_logits[idxs]
            self.rollouts_threshold_dict[cls] = np.percentile(this_class_robot_logits,
                                                self._rollout_drop_dict[cls] * 100.)
            wandb.log({"Rollouts/sample labeled to {}".format(cls): len(idxs),
                       "Rollouts/samples discarded {}".format(cls): np.sum(this_class_own_logits < self.threshold_dict[cls]),
                       "Rollouts/{}_threshold".format(cls): self.rollouts_threshold_dict[cls]
                       })
            
    def _cache_relabeling_info_for_ep(self, ep):
        agentview, eih, true_labels, actions = self.obtain_inputs_and_labels(ep)
        
        logits_labels, preds_labels, logits_all = self.classify_rollout(agentview, eih, true_labels, actions)

        del agentview, eih

        self.logits_labels_data.extend(logits_labels)
        self.preds_labels_data.extend(preds_labels)
        self.logits_all_data.extend(logits_all)
        self.true_labels_data.extend(true_labels)

    def get_relabled_values_for_ep(self, ep):

        agentview, eih, true_labels, actions = self.obtain_inputs_and_labels(ep)
        logits_labels, preds_labels, logits_all = self.classify_rollout(agentview, eih, true_labels, actions)

        remapping = {
            MODEL_DEMO: REAL_DEMO,
            MODEL_ROLLOUT: REAL_ROLLOUT,
            MODEL_INTV: REAL_INTV,
            MODEL_PREINTV: REAL_PREINTV,
        }

        if self._pseudo_labling: # pseudo labeling changing class label
            assert not self._label_rollout_only
            #assert self._relabel_strategy == "all"

            # Check for non-rollout classes, if logit is below lbl_cls_drop_thrshold. If yes, change to rollout
            # Check for rollout class. if logit is below rollout_accept_thrshold. If yes, stay at rollout
            for i in range(len(preds_labels)):
                true_lbl_ = int(true_labels[i])
                preds_label = int(preds_labels[i])
                pred_logit_ = logits_labels[i]
                logit_all_ = logits_all[i]

                if true_lbl_ == MODEL_ROLLOUT:
                    if pred_logit_ < self.rollouts_threshold_dict[preds_label]:
                        preds_labels[i] = MODEL_ROLLOUT
                else:
                    if pred_logit_ < self.threshold_dict[preds_label]: #and preds_label == true_lbl_:
                        preds_labels[i] = MODEL_ROLLOUT
        else:
            preds_labels = np.where(np.array(logits_labels) > self._cls_threshold, preds_labels, true_labels)
            if self._cls_threshold == 1:
                assert (np.array(preds_labels) == np.array(true_labels)).all()

        if self._label_rollout_only:
            for i in range(len(preds_labels)):
                true_labels_remap = remapping[true_labels[i]]
                if true_labels_remap != 0:
                    preds_labels[i] = true_labels[i]

        if self._relabel_strategy == "intv":
            for i in range(len(preds_labels)):
                pred = preds_labels[i]
                if pred != MODEL_INTV:
                    preds_labels[i] = true_labels[i]
        elif self._relabel_strategy == "intv_pre_intv":
            for i in range(len(preds_labels)):
                pred = preds_labels[i]
                if pred != MODEL_INTV and pred != MODEL_PREINTV:
                    preds_labels[i] = true_labels[i]

        pred_labels_new = []
        for pred in preds_labels:
            new_label = remapping[pred]
            pred_labels_new.append(new_label)

        true_labels_new = []
        for true_label in true_labels:
            new_label = remapping[true_label]
            true_labels_new.append(new_label)

        if not self._pseudo_labling and self._cls_threshold == 1:
            assert (np.array(pred_labels_new) == np.array(true_labels_new)).all()

        preintv_relabeled = self._data_override[ep][self._label_key]
        assert (np.array(true_labels_new) == np.array(preintv_relabeled)).all()

        assert len(pred_labels_new) == len(actions)

        return pred_labels_new

    def classify_rollout(self, agentview, eih, true_labels, actions):
        logits_labels = []
        preds_labels = []
        logits_all = []

        success_total = 0

        seq_len = self._seq_len

        for i in range(0, agentview.shape[1] - seq_len + 1):
            agentview_seq = agentview[:, i:i + seq_len]
            eih_seq = eih[:, i:i + seq_len]
            obs = {"agentview_image": agentview_seq,
                   "robot0_eye_in_hand_image": eih_seq}
            """
            obs = {"agentview_image": agentview_seq,
                   "eye_in_hand_image": eih_seq}
            """
            actions_seq = actions[i:i + seq_len, :]
            inputs = {"obs": obs, "actions": torch.from_numpy(actions_seq[None, :]).float().cuda()}

            true_label_seg = true_labels[i:i + seq_len]
            if self._traj_label_type == "middle":
                true_label = true_label_seg[seq_len // 2]
            elif self._traj_label_type == "last":
                true_label = true_label_seg[-1]
            else:
                assert NotImplementedError

            if self._debug:
                preds = 0
                logits_labels.append(0)
                success = True
            else:
                logits = self._classifier(inputs, "val")
                preds = torch.argmax(logits, 1).to(torch.float32).detach().item()
                logits = torch.nn.Softmax()(logits).detach().cpu()
                logits_all.append(logits)
                logit_label = logits.detach()[:, int(preds)].item()
                logits_labels.append(logit_label)

            preds_labels.append(preds)

        if self._traj_label_type == "middle":
            preds_labels = list(true_labels[:seq_len // 2]) + list(preds_labels) \
                           + list(true_labels[-seq_len // 2 - 1:])
            logits_labels = [1] * (seq_len // 2) + list(logits_labels) + [1] * (seq_len // 2 - 1)
            logits_all = [torch.ones(logits_all[0].shape)] * (seq_len // 2) \
                            + logits_all + \
                            [torch.ones(logits_all[0].shape)] * (seq_len // 2 - 1)

        elif self._traj_label_type == "last":
            preds_labels = list(true_labels[:seq_len - 1]) + list(preds_labels)
            logits_labels = [1] * (seq_len - 1) + list(logits_labels)
            logits_all = [torch.ones(logits_all[0].shape)] * (seq_len - 1) + list(logits_all)

        assert len(preds_labels) == agentview.shape[1]
        assert len(logits_all) == agentview.shape[1]

        return logits_labels, preds_labels, logits_all

    def obtain_inputs_and_labels(self, ep):
        data_orig = self.get_trajectory_at_index(
            demo_id=ep,
            obs_keys=["agentview_image", "robot0_eye_in_hand_image"]
        )  # TODO: change for coffee machine

        data = deepcopy(data_orig)

        agentview = data["obs"]["agentview_image"]
        eih = data["obs"]["robot0_eye_in_hand_image"]
    
        transpose_agent = "agentview_image" not in self.obs_keys
        agentview = self.process_image(agentview, transpose_agent)

        transpose_eih = "robot0_eye_in_hand_image" not in self.obs_keys
        eih = self.process_image(eih, transpose_eih)
        """

        data_orig = self.get_trajectory_at_index(
            demo_id=ep,
            obs_keys=["agentview_image", "eye_in_hand_image"]
        )  # TODO: change for coffee machine

        data = deepcopy(data_orig)

        agentview = data["obs"]["agentview_image"]
        eih = data["obs"]["eye_in_hand_image"]

        transpose_agent = "agentview_image" not in self.obs_keys
        agentview = self.process_image(agentview, transpose_agent)

        transpose_eih = "eye_in_hand_image" not in self.obs_keys
        eih = self.process_image(eih, transpose_eih)
        """
        assert agentview.shape[-3] == 3 and agentview.shape[-2] == agentview.shape[-1]
        assert eih.shape[-3] == 3 and eih.shape[-2] == eih.shape[-1]

        actions = data["actions"]

        # remapping intv labels to 0,1,2,3 as in model outputs
        true_labels_bef = data[self._label_key][()]
        true_labels = np.copy(true_labels_bef)

        if self._label_key == "intv_labels":
            for key in real_to_model:
                true_labels = np.where(true_labels_bef == key,
                                       torch.ones(size=true_labels.shape) * real_to_model[key],
                                       true_labels)
        return agentview, eih, true_labels, actions

    def process_image(self, img, transpose):
        image = torch.from_numpy(img)[None, :, :, :, :].cuda().float()
        if transpose:
            image = image.permute(0, 1, 4, 2, 3) / 255
        return image

    def get_trajectory_at_index(self, index=None, demo_id=None, obs_keys=None):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        assert (index is not None) + (demo_id is not None) == 1

        if demo_id is None:
            demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        if obs_keys is None:
            obs_keys = self.obs_keys

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta


class ClassifierRelabeledDataset(PreintvRelabeledDataset):
    def __init__(
            self,
            classifier_ckpt,
            relabeling_filter_key=None,
            label_key="intv_labels",
            label_rollout_only=True,
            relabel_strategy="intv",  # choose from "intv", "intv_and_pre_intv", "all"
            traj_label_type="middle",
            cls_threshold=0,
            debug=False,
            *args, **kwargs
    ):
        self._label_key = label_key
        super().__init__(*args, **kwargs)

        # DO NOT REMOVE THIS LINE. because we are relabeling, check we are not caching __getitem__ calls
        assert self.hdf5_cache_mode is not "all"

        # set relabeling key
        if relabeling_filter_key is None:
            relabeling_filter_key = self.filter_by_attribute

        # determine which demos should be relabled by classifier
        if relabeling_filter_key is not None:
            self.demos_relabeling = [elem.decode("utf-8") for elem in
                                     np.array(self.hdf5_file["mask/{}".format(relabeling_filter_key)][:])]
        else:
            self.demos_relabeling = list(self.hdf5_file["data"].keys())

        # for now let's assume no frame stacking for simplicity
        assert self.n_frame_stack == 1

        assert label_key in ["action_modes", "intv_labels"]
        self._label_key = label_key
        self._label_rollout_only = label_rollout_only
        self._debug = debug
        self._cls_threshold = cls_threshold

        if debug:
            self._classifier = None
            self._seq_len = 10
        else:
            self._classifier = torch.load(classifier_ckpt)
            self._seq_len = self._classifier.lstm_seq_length

        # cache of relabeled values that grows on demand
        self._relabeled_values_cache = dict()

        assert relabel_strategy in ["intv", "intv_pre_intv", "all"]
        self._relabel_strategy = relabel_strategy
        self._traj_label_type = traj_label_type

        samples_sum = 0
        samples_intv_sum = 0 

        print("Relabeling with classifier...")
        for ep in LogUtils.custom_tqdm(self.demos_relabeling):
            relabels = np.array(self.get_relabled_values_for_ep(ep))
            
            print("final label: ")
            print(relabels)
            samples_sum += len(relabels)
            samples_intv_sum += np.sum(relabels == 1)

            self._data_override[ep][self._label_key] = relabels
        
        print("total samples: ", samples_sum)
        print("intv samples: ", samples_intv_sum)
        print("intv ratio: ", samples_intv_sum / samples_sum)
        
        print("Done.")
        self._update_weights()

    def get_relabled_values_for_ep(self, ep):
        agentview, eih, true_labels_raw, actions = self.obtain_inputs_and_labels(ep)

        logits_labels, preds_labels = self.classify_rollout(agentview, eih, true_labels_raw, actions)
        #print(preds_labels)


        true_labels = true_labels_raw

        # remapping = {
        #     MODEL_DEMO: REAL_DEMO,
        #     MODEL_ROLLOUT: REAL_ROLLOUT,
        #     MODEL_INTV: REAL_INTV,
        #     MODEL_PREINTV: REAL_PREINTV,
        # }

        remapping = {
            REAL_DEMO: REAL_ROLLOUT, # TODO: Here treat demos as rollouts
            REAL_ROLLOUT: REAL_ROLLOUT,
            REAL_INTV: REAL_INTV,
            REAL_PREINTV: REAL_ROLLOUT,  # TODO: Here treat preintv as rollouts
        }
        # TODO: Assumes two class classification

        preds_labels = np.where(np.array(logits_labels) > self._cls_threshold, preds_labels, true_labels)

        if self._cls_threshold == 1:
            assert (np.array(preds_labels) == np.array(true_labels)).all()

        if self._label_rollout_only:
            for i in range(len(preds_labels)):
                true_labels_remap = remapping[true_labels[i]]
                if true_labels_remap != 0:
                    preds_labels[i] = true_labels[i]

        if self._relabel_strategy == "intv":
            for i in range(len(preds_labels)):
                pred = preds_labels[i]
                if pred != REAL_INTV: #MODEL_INTV:
                    preds_labels[i] = true_labels[i]
        elif self._relabel_strategy == "intv_pre_intv":
            for i in range(len(preds_labels)):
                pred = preds_labels[i]
                if pred != REAL_INTV and pred != REAL_PREINTV:
                    preds_labels[i] = true_labels[i]

        pred_labels_new = []
        for pred in preds_labels:
            new_label = remapping[pred]
            pred_labels_new.append(new_label)

        true_labels_new = []
        for true_label in true_labels:
            new_label = remapping[true_label]
            true_labels_new.append(new_label)

        if self._cls_threshold == 1:
            assert (np.array(pred_labels_new) == np.array(true_labels_new)).all()

        #preintv_relabeled = self._data_override[ep][self._label_key]
        #assert (np.array(true_labels_new) == np.array(preintv_relabeled)).all()

        assert len(pred_labels_new) == len(actions)

        return pred_labels_new

    def classify_rollout(self, agentview, eih, true_labels, actions):
        logits_labels = []
        preds_labels = []
        success_total = 0

        seq_len = self._seq_len

        for i in range(0, agentview.shape[1] - seq_len + 1):
            agentview_seq = agentview[:, i:i + seq_len]
            eih_seq = eih[:, i:i + seq_len]
            obs = {"agentview_image": agentview_seq,
                   "robot0_eye_in_hand_image": eih_seq}
            actions_seq = actions[i:i + seq_len, :]
            inputs = {"obs": obs, "actions": torch.from_numpy(actions_seq[None, :]).float().cuda()}

            true_label_seg = true_labels[i:i + seq_len]
            if self._traj_label_type == "middle":
                true_label = true_label_seg[seq_len // 2]
            elif self._traj_label_type == "last":
                true_label = true_label_seg[-1]
            else:
                assert NotImplementedError

            if self._debug:
                preds = 0
                logits_labels.append(0)
                success = True
            else:
                logits = self._classifier(inputs, "val")
                preds = torch.argmax(logits, 1).to(torch.float32).detach().item()
                logits = torch.nn.Softmax()(logits)
                logit_label = logits[:, int(preds)].item()
                logits_labels.append(logit_label)

            preds_labels.append(preds)

        if self._traj_label_type == "middle":
            preds_labels = list(true_labels[:seq_len // 2]) + list(preds_labels) \
                           + list(true_labels[-seq_len // 2 - 1:])
            logits_labels = [1] * (seq_len // 2) + list(logits_labels) + [1] * (seq_len // 2 - 1)
        elif self._traj_label_type == "last":
            #preds_labels = list(true_labels[:seq_len - 1]) + list(preds_labels)
            preds_labels = [0] * (seq_len - 1) + list(preds_labels) # hack for now
            logits_labels = [1] * (seq_len - 1) + list(logits_labels)

        assert len(preds_labels) == agentview.shape[1]

        return logits_labels, preds_labels

    def obtain_inputs_and_labels(self, ep):
        data_orig = self.get_trajectory_at_index(
            demo_id=ep,
            obs_keys=["agentview_image", "robot0_eye_in_hand_image"]
        )  # TODO: change for coffee machine

        data = deepcopy(data_orig)

        agentview = data["obs"]["agentview_image"]
        eih = data["obs"]["robot0_eye_in_hand_image"]

        transpose_agent = "agentview_image" not in self.obs_keys
        agentview = self.process_image(agentview, transpose_agent)

        transpose_eih = "robot0_eye_in_hand_image" not in self.obs_keys
        eih = self.process_image(eih, transpose_eih)

        assert agentview.shape[-3] == 3 and agentview.shape[-2] == agentview.shape[-1]
        assert eih.shape[-3] == 3 and eih.shape[-2] == eih.shape[-1]

        actions = data["actions"]

        # remapping intv labels to 0,1,2,3 as in model outputs
        true_labels_bef = data[self._label_key][()]
        true_labels = np.copy(true_labels_bef)

        if self._label_key == "intv_labels":
            for key in real_to_model:
                true_labels = np.where(true_labels_bef == key,
                                       torch.ones(size=true_labels.shape) * real_to_model[key],
                                       true_labels)
        return agentview, eih, true_labels, actions

    def process_image(self, img, transpose):
        image = torch.from_numpy(img)[None, :, :, :, :].cuda().float()
        if transpose:
            image = image.permute(0, 1, 4, 2, 3) / 255
        return image

    def get_trajectory_at_index(self, index=None, demo_id=None, obs_keys=None):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        assert (index is not None) + (demo_id is not None) == 1

        if demo_id is None:
            demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        if obs_keys is None:
            obs_keys = self.obs_keys

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta
