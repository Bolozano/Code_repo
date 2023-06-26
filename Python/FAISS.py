train_array=torch.cat([temp.unsqueeze(0) for temp in list(data_store.keys())],dim=0).numpy()
np.save('/GPFS/data/yutongmeng/VCR/knn_datastore/train_array.npy',pp)


import faiss
class args:
    dimension=768
    ncentroids=4096
    code_size=64
    dstore_size=7130949
    test_len=904109

quantizer = faiss.IndexFlatL2(args.dimension)#faiss.IndexFlatL2(args.dimension)
index = faiss.IndexIVFPQ(quantizer, args.dimension,
        args.ncentroids, args.code_size, 8)
train_array=np.memmap('/GPFS/data/yutongmeng/VCR/knn_datastore/train_array.bin', dtype=np.float32, mode='r', shape=(args.dstore_size,args.dimension))
test_array=np.memmap('/GPFS/data/yutongmeng/VCR/knn_datastore/test_array.bin', dtype=np.float32, mode='r', shape=(args.test_len,args.dimension))

# gpu_index.train(train_array)
index.train(train_array)
index.add_with_ids(train_array,np.arange(train_array.shape[0]))
faiss.write_index(index,'/GPFS/data/yutongmeng/VCR/knn_datastore/index.trained')
index = faiss.read_index('/GPFS/data/yutongmeng/VCR/knn_datastore/index.trained')
