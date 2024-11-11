for m in 30 50 100 150 200
do
    python merge_datasets.py --dataset /scratch/cluster/huihanl/robomimic-hitl/datasets/huihan_hitl/0518_round0_only.hdf5 /scratch/cluster/huihanl/robomimic-hitl/datasets/huihan_hitl/round1_600_200trajs.hdf5 --output_dataset /scratch/cluster/huihanl/robomimic-hitl/datasets/huihan_hitl/0804_50_demos_${m}_rollouts.hdf5 --num_lst 50 $m
done

