# BGU-GCN: Enhanced Graph Convolutional Networks for Collaborative Filtering

This repository contains the implementation of BGU-GCN, an enhanced Graph Convolutional Network (GCN) architecture for collaborative filtering in recommender systems. BGU-GCN introduces several key optimizations to improve recommendation accuracy and training efficiency compared to state-of-the-art methods.

## Features

- Weighted loss function to prioritize informative training samples
- Adaptive margins to learn fine-grained user-item relationships
- Similarity-based embedding initialization for faster convergence
- Strategic edge removal techniques (bridge removal and noisy edge removal) to reduce computational complexity and improve embedding quality
- Comprehensive evaluation on benchmark datasets (Amazon-Book and Gowalla)

## Requirements

- Python 3.x
- PyTorch
- NumPy
- SciPy
- scikit-learn

## Usage

1. Clone the repository:
'''python

git clone https://github.com/OrKatz7/BGUGCN.git
cd BGUGCN
'''
3. Prepare the dataset:
- Download the Amazon-Book and Gowalla datasets
- Preprocess the data and save it in the appropriate format

3. Train the BGU-GCN model:

   python train.py --dataset amazon-book --epochs 100 --batch-size 1024 --lr 0.001

4. Evaluate the trained model:

python evaluate.py --dataset amazon-book --model-path /path/to/trained/model

## Results

BGU-GCN outperforms existing state-of-the-art collaborative filtering methods, including LightGCN and UltraGCN, on the Amazon-Book and Gowalla datasets:

- On Amazon-Book, BGU-GCN achieves a 66.91% increase in Recall@20 and a 76.50% increase in NDCG@20 compared to LightGCN, while converging 12x faster.
- On Gowalla, BGU-GCN provides a 2.78% improvement in Recall@20 and 2.64% in NDCG@20 compared to LightGCN, converging in less than half the epochs.

The model's faster convergence makes it well-suited for real-world recommendation scenarios involving massive datasets.

## Citation

If you find this work useful for your research, please consider citing our paper:

@inproceedings{arbili2023bgu-gcn,
title={BGU-GCN: Enhanced Graph Convolutional Networks for Collaborative Filtering},
author={Arbili, Ofir and Katz, Or},
booktitle={Software and Information Systems Engineering (SISE) at Ben Gurion University (BGU)},
year={2023}
}

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or feedback, please contact the authors:
- Ofir Arbili (arbili@post.bgu.ac.il)
- Or Katz (kaor@post.bgu.ac.il)
