python hitl/truncate_successful_ends.py $1.hdf5 5
python hitl/add_hc_weights.py --dataset $1_trunc_5.hdf5 --output_dataset $1_intv_labels.hdf5
        
        
