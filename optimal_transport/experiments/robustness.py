from ._experiment import Experiment
from ..models._ot import _OT
from ..classifiers import KNN

from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import os


class Robustness(Experiment):
    def __init__(
        self,
        model: Dict[int, _OT],
        exp_name: str,
        log_dir: str,
    ):
        super().__init__(model, exp_name, log_dir)
    
    def run(
        self, xs: np.ndarray, xt: np.ndarray, ys: np.ndarray, yt: np.ndarray, 
        model: _OT, n_neighbors: int = 1, **kwargs
    ) -> float:
        def accuracy(preds, labels):
            return np.sum(preds == labels) / labels.shape[0]
        
        a = np.ones(xs.shape[0]) / xs.shape[0]
        b = np.ones(xt.shape[0]) / xt.shape[0]
        knn = KNN(n_neighbors)
        knn.fit(xt, yt)
        model.fit(xs, xt, a, b, **kwargs)
        xs_t = model.transport(xs, xt)
        ys_pred = knn.predict(xs_t)
        return accuracy(ys_pred, ys)

    def plot(
        self, x_axis: str, y_axis: str,
        save_fig: bool = True, **kwargs
    ):
        plt.figure(figsize=(12, 8))
        for algo, record in self.record_[self.exp_name].items():
            plt.plot(record[x_axis], record[y_axis], label=algo)
        
        plt.title(self.exp_name)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend()
        plt.grid(True)

        if save_fig:
            plt.savefig(os.path.join(self.log_dir, f"{self.cur_time}.png"), dpi=300)
        plt.show()

    @classmethod
    def gaussMixture_meanRandom_covWishart(
        cls,
        n: int,
        d: int,
        k: int, # number of clusters
        d_proj: Optional[int] = None,
        means: Optional[np.ndarray] = None,
        covs: Optional[Union[np.ndarray, List]] = None,
        org_noise_level: float = 0,
        dim_noise_level: float = 0,
        cov_scale: float = 1,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        assert n % k == 0, "Each clusters should have equal number of samples."
        if d_proj is None:
            d_proj = d

        # Original data ................................
        # generate random means for each cluster
        if means is None:
            means = np.random.randn(k, d)

        # sample covariance matrices from a Wishart distribution
        if covs is None:
            covs = []
            for _ in range(k):
                cov = np.eye(d)
                covs.append(cov)
            covs = np.array(covs)
        covs = cov_scale * covs

        # generate data points for each cluster
        X, y, K = [], [], []
        for i in range(k):
            Xi = np.random.multivariate_normal(means[i], covs[i], size=(n // k - 1))
            Xi = np.concatenate((means[i][None, :], Xi))
            X.append(Xi) # (n, d)
            y.append(i * np.ones(n // k))
            K.append(i * (n // k))
        X, y = np.concatenate(X), np.concatenate(y)

        # Pertubations ................................
        # add gaussian noise to each component
        if org_noise_level > 0:
            X = Robustness.add_noise_plane(X, y, means, noise_level=org_noise_level)

        # noisely project data points to a higher-dimensional subspace
        if d_proj > d:
            X = Robustness.add_noise_dim(X, d_proj-d, noise_level=dim_noise_level)
        
        return X, y, K # <-- K ~ centroid indices
    
    @classmethod
    def add_noise_dim(cls, X: np.ndarray, n_dims: int, 
            noise_level: float = 1) -> np.ndarray:
        if n_dims == 0:
            return X
        X_ = X.copy()
        noise = np.random.normal(scale=noise_level, size=(X.shape[0], n_dims))
        X_ = np.concatenate([X, noise], axis=1) # n x (d + d_noise)
        return X_
    
    @classmethod
    def add_noise_plane(cls, X: np.ndarray, y:np.ndarray, means: np.ndarray, 
            noise_level: float = 1) -> np.ndarray:
        if noise_level == 0:
            return X
        inds = np.arange(X.shape[0])
        np.random.shuffle(inds)
        inds = inds[:int(noise_level * X.shape[0])]
        X_ = X.copy()
        X_[inds] = X[inds] + np.random.normal(
            scale=np.sqrt(0.5 * np.square(means[y[inds].astype("int64")])), 
            size=(len(inds), X.shape[1])
        )
        return X_


class Dimensionality(Robustness):
    def __init__(
        self,
        model: Dict[int, _OT],
        log_dir: str,
    ):
        super().__init__(model, exp_name="dimensionality", log_dir=log_dir)

    def __call__(
        self,
        noise_level: float = 1,
        max_projected_dim: int = 100,
        freq_projected_dim: int = 5,
        hyperplane_dim: int = 5,
        n_components: int = 4,
        cluster_samples: int = 100,
        n_keypoints: Optional[int] = 4,
        **kwargs
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"dimension": [], "accuracy": [], "runtime": []} for model_id in self.model}
        assert (max_projected_dim - hyperplane_dim) % freq_projected_dim == 0
        
        sample_size = n_components * cluster_samples
        Xs, ys, Ks = Robustness.gaussMixture_meanRandom_covWishart(sample_size, hyperplane_dim, n_components, d_proj=hyperplane_dim, **kwargs)
        Xt, yt, Kt = Robustness.gaussMixture_meanRandom_covWishart(sample_size, hyperplane_dim, n_components, d_proj=hyperplane_dim, **kwargs)
        K = [(Ks[i], Kt[i]) for i in range(len(Ks))][:n_keypoints]

        for prj_dim in range(hyperplane_dim, max_projected_dim+1, freq_projected_dim):
            Xs_ = Robustness.add_noise_dim(Xs, prj_dim-hyperplane_dim, noise_level=noise_level)
            Xt_ = Robustness.add_noise_dim(Xt, prj_dim-hyperplane_dim, noise_level=noise_level)

            for model_id, model in self.model.items():
                start = time.time()
                acc = self.run(Xs_, Xt_, ys, yt, model, K=K)
                self.record_[self.exp_name][model_id]["dimension"].append(prj_dim)
                self.record_[self.exp_name][model_id]["accuracy"].append(acc)
                self.record_[self.exp_name][model_id]["runtime"].append(time.time() - start)
            
            acc_log = {model_id: self.record_[self.exp_name][model_id]["accuracy"][-1] for model_id in self.record_[self.exp_name]}
            runtime_log = {model_id: self.record_[self.exp_name][model_id]["runtime"][-1] for model_id in self.record_[self.exp_name]}
            self.checkpoint()
            self.logger.info(f"Dimension: {prj_dim}, Accuracy: {acc_log}, Runtime: {runtime_log}")

        self.plot(x_axis="dimension", y_axis="accuracy")
        return self.record_[self.exp_name]
    

class OutlierRate(Robustness):
    def __init__(
        self,
        model: Dict[int, _OT],
        log_dir: str,
    ):
        super().__init__(model, exp_name="outlier_rate", log_dir=log_dir)

    def __call__(
        self,
        max_noise_ratio: float = 1,
        freq_noise_ratio: float = 0.1,
        hyperplane_dim: int = 30,
        cluster_samples: int = 100,
        n_keypoints: Optional[int] = 4,
        n_components: int = 4,
        **kwargs
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"ratio": [], "accuracy": [], "runtime": []} for model_id in self.model}

        sample_size = n_components * cluster_samples
        Xs, ys, Ks = Robustness.gaussMixture_meanRandom_covWishart(sample_size, hyperplane_dim, n_components, d_proj=hyperplane_dim, **kwargs)
        Xt, yt, Kt = Robustness.gaussMixture_meanRandom_covWishart(sample_size, hyperplane_dim, n_components, d_proj=hyperplane_dim, **kwargs)
        K = [(Ks[i], Kt[i]) for i in range(len(Ks))][:n_keypoints]

        for noise_ratio in [freq_noise_ratio * i for i in range(int(max_noise_ratio // freq_noise_ratio) + 1)]:
            Xs_ = Robustness.add_noise_plane(Xs, ys, Xs[Ks][np.argsort(ys[Ks])], noise_level=noise_ratio)
            Xt_ = Robustness.add_noise_plane(Xt, yt, Xt[Kt][np.argsort(ys[Ks])], noise_level=noise_ratio)

            for model_id, model in self.model.items():
                start = time.time()
                acc = self.run(Xs_, Xt_, ys, yt, model, K=K)
                self.record_[self.exp_name][model_id]["ratio"].append(noise_ratio)
                self.record_[self.exp_name][model_id]["accuracy"].append(acc)
                self.record_[self.exp_name][model_id]["runtime"].append(time.time() - start)

            acc_log = {model_id: self.record_[self.exp_name][model_id]["accuracy"][-1] for model_id in self.record_[self.exp_name]}
            runtime_log = {model_id: self.record_[self.exp_name][model_id]["runtime"][-1] for model_id in self.record_[self.exp_name]}
            self.checkpoint()
            self.logger.info(f"Noise ratio: {noise_ratio}, Accuracy: {acc_log}, Runtime: {runtime_log}")

        self.plot(x_axis="ratio", y_axis="accuracy")
        return self.record_[self.exp_name]
    

class ClusterMismatch(Robustness):
    def __init__(
        self,
        model: Dict[int, _OT],
        log_dir: str,
    ):
        super().__init__(model, exp_name="cluster_mismatch", log_dir=log_dir)

    def __call__(
        self,
        min_source_components: int = 2,
        freq_components: int = 1,
        target_components: int = 10,
        hyperplane_dim: int = 30,
        cluster_samples: int = 100,
        n_keypoints: Optional[int] = 10,
        src_properties: Optional[Dict] = None,
        trg_properties: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"cluster": [], "accuracy": [], "runtime": []} for model_id in self.model}
        assert (target_components - min_source_components) % freq_components == 0

        Xs, ys, Ks = Robustness.gaussMixture_meanRandom_covWishart(target_components * cluster_samples, hyperplane_dim, target_components, d_proj=hyperplane_dim, **src_properties, **kwargs)
        Xt, yt, Kt = Robustness.gaussMixture_meanRandom_covWishart(target_components * cluster_samples, hyperplane_dim, target_components, d_proj=hyperplane_dim, **trg_properties, **kwargs)
        K = [(Ks[i], Kt[i]) for i in range(len(Ks))][:n_keypoints]

        for n_components in range(min_source_components, target_components+1, freq_components):
            mask = np.in1d(ys, [i for i in range(n_components)])
            Xs_, ys_ = Xs[mask], ys[mask]
            K_ = K[:n_components]

            for model_id, model in self.model.items():
                start = time.time()
                acc = self.run(Xs_, Xt, ys_, yt, model, K=K_, **kwargs)
                self.record_[self.exp_name][model_id]["cluster"].append(n_components)
                self.record_[self.exp_name][model_id]["accuracy"].append(acc)
                self.record_[self.exp_name][model_id]["runtime"].append(time.time() - start)

            acc_log = {model_id: self.record_[self.exp_name][model_id]["accuracy"][-1] for model_id in self.record_[self.exp_name]}
            runtime_log = {model_id: self.record_[self.exp_name][model_id]["runtime"][-1] for model_id in self.record_[self.exp_name]}
            self.checkpoint()
            self.logger.info(f"Number of source components: {n_components}, Accuracy: {acc_log}, Runtime: {runtime_log}")

        self.plot(x_axis="cluster", y_axis="accuracy")
        return self.record_[self.exp_name]