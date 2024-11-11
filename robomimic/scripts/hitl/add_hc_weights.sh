for m in 0804_30_demos_100_rollouts.hdf5  0804_30_demos_30_rollouts.hdf5   0804_50_demos_150_rollouts.hdf5  0804_50_demos_50_rollouts.hdf5 0804_30_demos_150_rollouts.hdf5  0804_30_demos_50_rollouts.hdf5   0804_50_demos_200_rollouts.hdf5 0804_30_demos_200_rollouts.hdf5  0804_50_demos_100_rollouts.hdf5  0804_50_demos_30_rollouts.hdf5

do
    python add_hc_weights.py --dataset /scratch/cluster/huihanl/robomimic-hitl/datasets/huihan_hitl/$m
done

