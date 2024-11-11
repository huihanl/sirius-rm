import gdown, os
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

gdown.download('https://drive.google.com/uc?export=download&id=1fLXJWS69C8JePNKXGpAK55Wt7duYdFeS', os.path.join(data_dir,'square_original.hdf5') , quiet=False) 
gdown.download('https://drive.google.com/uc?export=download&id=1aju_soVuhWlXbil9loI2Zdkv_GIlxOtj', os.path.join(data_dir,'square_relabeled_scale1.hdf5') , quiet=False)
gdown.download('https://drive.google.com/uc?export=download&id=1C6Rq5yKh9ZRVFTFfMddNF-M_lK3rUwbr', os.path.join(data_dir,'square_relabeled_scale2.hdf5') , quiet=False)

gdown.download('https://drive.google.com/uc?export=download&id=1_Rj6Xn1T0RwqyzcqHDj_84g4vLFk1ZiX', os.path.join(data_dir,'coffee_original.hdf5') , quiet=False) 
gdown.download('https://drive.google.com/uc?export=download&id=1-bk5M_Lpu9_RcbKBfDAGrlYEtoZaDy5q', os.path.join(data_dir,'coffee_relabeled_scale1.hdf5') , quiet=False)
