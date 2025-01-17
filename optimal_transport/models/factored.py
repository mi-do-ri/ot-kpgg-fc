from ._ot import _OT
from ..utils import Distance, SquaredEuclidean, JSDiv, softmax
from sklearn.metrics import pairwise_distances_argmin_min
from typing import Optional, List, Tuple
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
import ot
from scipy import stats
from scipy.sparse import random

class KeypointFOT(_OT):
    def __init__(
        self,
        ys: np.ndarray,
        distance: Distance = SquaredEuclidean,
        similarity: Distance = JSDiv,
        n_free_anchors: Optional[int] = None,
        sinkhorn_reg: float = 0.01, 
        temperature: float = 0.1, 
        div_term: float = 1e-10, 
        alpha: float = 0.5,
        stop_thr: float = 1e-5, 
        max_iters: int = 100,
        n_clusters: int = 3,
        fused: bool = True
    ):
        super().__init__(distance)
        self.ys = ys
        self.sim_fn = similarity
        self.dist_fn = distance
        self.k = n_free_anchors
        self.eps = sinkhorn_reg
        self.rho = temperature
        self.div_term = div_term
        self.stop_thr = stop_thr
        self.max_iters = max_iters
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.fused = fused

        self.Pa_: Optional[np.ndarray] = None
        self.Pb_: Optional[np.ndarray] = None
        self.z_: Optional[np.ndarray] = None
        

    def fit(
        self,
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        K: List[Tuple],
        **kwargs,
    ) -> "KeypointFOT":
        vs=xs.sum(axis=0) # set the weights on the features
        vs/=vs.sum()
        vt=xt.sum(axis=0)
        vt/=vt.sum()
        
        # z, h = self._init_anchors(xs, self.n_clusters, self.k, len(K))
        z, h, L = self._init_anchors(xs, self.ys, self.n_clusters, self.k, len(K))
        print("inital z: ", z)
        # I, L, J = self._init_keypoint_inds(K)
        I, J = self._init_keypoint_inds(K)
        Ms, Mt = self._init_masks(xs, z, xt, I, L, J)

        self.z_ = z
        Cs = self.dist_fn(xs, xs)
        Cs = Cs / (Cs.max() + self.div_term)
        Ct = self.dist_fn(xt, xt)
        Ct = Ct / (Ct.max() + self.div_term)
         
        for i in range(self.max_iters):
            Cz = self.dist_fn(z, z)
            Cz = Cz / (Cz.max() + self.div_term)
            
            Gs = self._guide_matrix(Cs, Cz, I, L)
            Gt = self._guide_matrix(Cz, Ct, L, J)
            
            Ps = self._update_plans(Cs, Cz, a, h, Gs, Ms)
            Pt = self._update_plans(Cz, Ct, h, b, Gt, Mt)
            # P = Ps.dot(Pt)
            # Pv = self._update_plan_feat(xs, xt, vs, vt, a, b, P)
            Pv = np.ones((xs.shape[1], xt.shape[1])) / (xs.shape[1] * xt.shape[1])
            print("Pv: ", Pv)
            z = self._update_anchors(xs, xt, Ps, Pt, Pv)
            #print("z: ", z)

            err = np.linalg.norm(z - self.z_)
            self.z_ = z
    
            if err <= self.stop_thr:
                print(f"Threshold reached at iteration {i}")
                break
        print(z)
        self.Pa_ = Ps
        self.Pb_ = Pt
        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        n = xs.shape[0]
        m = xt.shape[0]
        assert (self.Pa_ is not None) and (self.Pb_ is not None), "Should run fit() before mapping"
        
        Cx = self.Pa_.T.dot(xs) / (self.Pa_.T.dot(np.ones((n, 1))) + self.div_term)
        Cy = self.Pb_.dot(xt) / (self.Pb_.dot(np.ones((m, 1))) + self.div_term)
        return xs + np.dot(self.Pa_ / np.sum(self.Pa_, axis=1).reshape([n, 1]), Cy - Cx)


    def _init_keypoint_inds(
        self,
        K: List[Tuple]
    ) -> Tuple[np.ndarray]:
        I = np.array([pair[0] for pair in K])
        J = np.array([pair[1] for pair in K])
        # L = np.arange(len(K))
        return I, J

    def _init_anchors(
        self, 
        x: np.ndarray,
        y: np.ndarray,
        n_clusters: int,
        n_free_anchors: int,
        n_keypoints: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = (int)(n_keypoints + n_free_anchors) // n_clusters
        # Z = []
        # for i in range(T): 
        #     model = KMeans(n_clusters=n_clusters)
        #     model.fit(x)
        #     z = model.cluster_centers_
        #     Z.append(z)

        unique_labels = np.unique(y)
        selected_centers = []

        for label in unique_labels:
            # Lấy chỉ mục của các điểm thuộc class hiện tại
            indices = np.where(y == label)[0]

            # Tính trung bình của các điểm trong class
            center = np.mean(x[indices], axis=0)

            # Tính khoảng cách giữa mỗi điểm và trung tâm
            distances = pairwise_distances_argmin_min(x[indices], [center])[1]

            # Chọn 2 điểm có khoảng cách nhỏ nhất
            selected_indices = indices[np.argsort(distances)[:T]]
            selected_centers.extend(selected_indices)

        Z = x[selected_centers]
        L = [i * T + 1 for i in range(len(Z)//T)]
        # Z = np.vstack(np.array(Z))
        h = np.ones(len(Z)) / (len(Z))
        return Z, h, L
    
    def _init_masks(
        self,
        xs: np.ndarray, z: np.ndarray, xt: np.ndarray,
        I: np.ndarray, L: np.ndarray, J: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Ms = self._guide_mask(xs, z, I, L)
        Mt = self._guide_mask(z, xt, L, J)
        return Ms, Mt
    
    def _init_matrix(
        self,
        Cx: np.ndarray, Cy: np.ndarray,
        p: np.ndarray, q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        def fx(a):
            return (a ** 2)
        def fy(b):
            return (b ** 2)
        def hx(a):
            return a
        def hy(b):
            return (b * 2)
        
        constCx = np.dot(np.dot(fx(Cx), p.reshape(-1, 1)),
                            np.ones(len(q)).reshape(1, -1))
        constCy = np.dot(np.ones(len(p)).reshape(-1, 1),
                            np.dot(q.reshape(1, -1), fy(Cy).T))
        
        constC = constCx + constCy
        hCx = hx(Cx)
        hCy = hy(Cy)
        
        return constC, hCx, hCy
    
    def _init_matrix_feat(
        self,
        xs: np.ndarray, xt: np.ndarray,
        vs: np.ndarray, vt: np.ndarray,
    )-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def f1(a):
            return (a ** 2)

        def f2(b):
            return (b ** 2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b

        constC1 = np.dot(np.dot(f1(xs), vs.reshape(-1, 1)),
                        np.ones(f1(xt).shape[0]).reshape(1, -1))
        constC2 = np.dot(np.ones(f1(xs).shape[0]).reshape(-1, 1),
                        np.dot(vt.reshape(1, -1), f2(xt).T))

        constC = constC1 + constC2
        hX1 = h1(xs)
        hX2 = h2(xt)

        return constC, hX1, hX2
    
    def _product(self, constC, hCx, hCy, T):
        A = -np.dot(np.dot(hCx, T), (hCy.T))
        tens = constC + A
        return tens
    
    def _gwloss(self, constC, hCx, hCy, T):
        tens = self._product(constC, hCx, hCy, T)
        return np.sum(tens * T)

    def _gwggrad(self, constC, hCx, hCy, T):
        return 2 * self._product(constC, hCx, hCy, T)
    
    def _update_anchors(
        self, 
        xs: np.ndarray, xt: np.ndarray,
        Ps: np.ndarray, 
        Pt: np.ndarray,
        Pv: np.ndarray
    ) -> np.ndarray:
        assert self.z_ is not None, "_init_anchors() did not run properly."
        z = 0.5 * (np.matmul((Ps).T, xs) + np.matmul(np.matmul(Pt, xt), Pv.T)) * len(self.z_)
        # z = 0.5 * (np.matmul((Ps).T, xs) + np.matmul(Pt, xt)) * len(self.z_)
        return z
    
    def _update_plan_feat(
        self,
        xs: np.ndarray, xt: np.ndarray,
        vs: np.ndarray, vt: np.ndarray,
        p: np.ndarray, q: np.ndarray,
        P: np.ndarray,
        random_init=False, log=False, verbose=False
    ) -> np.array:
        if vs is None:
            vs = np.ones(xs.shape[1]) / xs.shape[1]
        if vt is None:
            vt = np.ones(xt.shape[1]) / xt.shape[1]
        
        if not random_init:
            Pv = np.ones((xs.shape[1], xt.shape[1])) / (xs.shape[1] * xt.shape[1])  # is (d,d')
        else:
            Pv = self.random_gamma_init(vs,vt)
        
        constC, hC1, hC2 = self._init_matrix_feat(xs.T, xt.T, p, q)

        cost = np.inf
        log_out = {}
        log_out['cost'] = []
        n_iter = 1
        cost_old = cost
        
        for i in range(n_iter):
            Pv_old = Pv
            M = constC - (hC1.dot(P)).dot(hC2.T)
            M = M / (M.max() + self.div_term)
            Pv = ot.sinkhorn(vs,vt, M, self.eps)
            
            delta = np.linalg.norm(Pv - Pv_old)
            cost = np.sum(M * Pv)
            
            if log:
                log_out['cost'].append(cost)
                
            if verbose:
                print('Delta: {0}  Loss: {1}'.format(delta, cost))

            if delta < 1e-16 or np.abs(cost_old - cost) < 1e-7:
                if verbose:
                    print('converged at iter ', i)
                break
            
            cost_old = cost
            # if log:
            #     return Pv, log_out
            # else:
            #     return Pv
        return Pv
            
        
    def random_gamma_init(self,p,q, **kwargs):
        """ Returns random coupling matrix with marginal p,q
        """
        rvs=stats.beta(1e-1,1e-1).rvs
        S=random(len(p), len(q), density=1, data_rvs=rvs)
        return self._sinkhorn_scaling(p,q,S.A, **kwargs)

    def _sinkhorn_scaling(self,a,b,K,numItermax=1000, stopThr=1e-9, verbose=False,log=False,always_raise=False, **kwargs):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        # init data
        Nini = len(a)
        Nfin = len(b)

        if len(b.shape) > 1:
            nbb = b.shape[1]
        else:
            nbb = 0

        if log:
            log = {'err': []}

        # we assume that no distances are null except those of the diagonal of
        # distances
        if nbb:
            u = np.ones((Nini, nbb)) / Nini
            v = np.ones((Nfin, nbb)) / Nfin
        else:
            u = np.ones(Nini) / Nini
            v = np.ones(Nfin) / Nfin

        # print(reg)
        # print(np.min(K))

        Kp = (1 / a).reshape(-1, 1) * K
        cpt = 0
        err = 1
        while (err > stopThr and cpt < numItermax):
            uprev = u
            vprev = v
            KtransposeU = np.dot(K.T, u)
            v = np.divide(b, KtransposeU)
            u = 1. / np.dot(Kp, v)

            zero_in_transp=np.any(KtransposeU == 0)
            nan_in_dual= np.any(np.isnan(u)) or np.any(np.isnan(v))
            inf_in_dual=np.any(np.isinf(u)) or np.any(np.isinf(v))
            if zero_in_transp or nan_in_dual or inf_in_dual:
                # we have reached the machine precision
                # come back to previous solution and quit loop
                print('Warning: numerical errors at iteration in sinkhorn_scaling', cpt)
                #if zero_in_transp:
                    #print('Zero in transp : ',KtransposeU)
                #if nan_in_dual:
                    #print('Nan in dual')
                    #print('u : ',u)
                    #print('v : ',v)
                    #print('KtransposeU ',KtransposeU)
                    #print('K ',K)
                    #print('M ',M)

                #    if always_raise:
                #        raise NanInDualError
                #if inf_in_dual:
                #    print('Inf in dual')
                u = uprev
                v = vprev

                break
            if cpt % 10 == 0:
                # we can speed up the process by checking for the error only all
                # the 10th iterations
                if nbb:
                    err = np.sum((u - uprev)**2) / np.sum((u)**2) + \
                        np.sum((v - vprev)**2) / np.sum((v)**2)
                else:
                    transp = u.reshape(-1, 1) * (K * v)
                    err = np.linalg.norm((np.sum(transp, axis=0) - b))**2
                if log:
                    log['err'].append(err)

                if verbose:
                    if cpt % 200 == 0:
                        print(
                            '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))
            cpt = cpt + 1
        if log:
            log['u'] = u
            log['v'] = v

        if nbb:  # return only loss
            res = np.zeros((nbb))
            for i in range(nbb):
                res[i] = np.sum(
                    u[:, i].reshape((-1, 1)) * K * v[:, i].reshape((1, -1)) * M)
            if log:
                return res, log
            else:
                return res

        else:  # return OT matrix

            if log:
                return u.reshape((-1, 1)) * K * v.reshape((1, -1)), log
            else:
                return u.reshape((-1, 1)) * K * v.reshape((1, -1))
    
    def _update_plans(
        self,
        Cx: np.ndarray, Cy: np.ndarray,
        p: np.ndarray, q: np.ndarray,
        G: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        G = G / (G.max() + self.div_term)
        
        constC, hCx, hCy = self._init_matrix(Cx, Cy, p, q)
        G0 = p.reshape(-1, 1) * q.reshape(1, -1)
        
        def f(G):
          return self._gwloss(constC, hCx, hCy, G)
        
        def df(G):
            return self._gwggrad(constC, hCx, hCy, G)
        
        P, f_val = self._cg(p, q, Cx, Cy, constC, f, df, G0, mask, G)
        self.f_val = f_val
        
        return P
    
    def _cg(
        self,
        p: np.ndarray, q: np.ndarray,
        Cx: np.ndarray, Cy: np.ndarray,
        constC: np.ndarray, f, df, 
        G0: np.ndarray, mask: np.ndarray,
        G: np.ndarray
    ) -> np.ndarray:
        numItermax = 5
        loop = 1
        
        def cost(G0):
            if not self.fused:
                return f(G0)
            else:
                return self.alpha * f(G0) + (1.0 - self.alpha) * np.sum(mask * G * G0)
        
        def cost_mask(G0):
            return cost(mask * G0)
        
        if mask is None:
            f_val = cost(G0)
        else:
            f_val = cost_mask(G0)
            
        it = 0
        
        while loop:
            it += 1
            old_fval = f_val
            
            if mask is None:
                dfG = df(G0)
            else:
                dfG = df(mask * G0)
            
            if self.fused: 
                M = self.alpha * dfG + (1.0 - self.alpha) * mask * G
            else:
                M = dfG
                
            M += M.min()
            
            Gc = self._sinkhorn_log_domain(p, q, M, mask)
            
            deltaG = Gc - G0
            
            if mask is None:
                alpha1, f_val = self._solve_linesearch(cost, G0, deltaG, Cx, Cy, constC)
            else:
                alpha1, f_val = self._solve_linesearch(cost, mask*G0, mask*deltaG, Cx, Cy, constC)
            G0 += alpha1 * deltaG
            
            if it >= numItermax:
                loop = 0
            
            abs_delta_fval = abs(f_val - old_fval)
            relative_delta_fval = abs_delta_fval / abs(f_val)
            if relative_delta_fval < self.stop_thr or abs_delta_fval < self.stop_thr:
                loop = 0
        
        return G0, f_val
    
    def _solve_linesearch(
        self,
        cost, G: np.ndarray, deltaG: np.ndarray, 
        Cx: np.array, Cy: np.ndarray, 
        constC: np.ndarray
    ):
        dotx = np.dot(Cx, deltaG)
        dotxy = np.dot(dotx, Cy)
        a = -2 * np.sum(dotxy * deltaG)
        b = np.sum(constC * deltaG) - 2 * (np.sum(dotxy * G) + np.sum(np.dot(np.dot(Cx, G), Cy) * deltaG))
        c = cost(G)
        
        alpha = self._solve_1d_linesearch_quad(a, b, c)
        f_val = cost(G + alpha * deltaG)
        
        return alpha, f_val
    
    def _solve_1d_linesearch_quad(
        self,
        a, b, c
    ):
        f0 = c
        df0 = b
        f1 = a + f0 + df0

        if a > 0:  # convex
            minimum = min(1, max(0, -b/2.0 * a))
            return minimum
        else:  # non convex
            if f0 > f1:
                return 1
            else:
                return 0
    
    def _sinkhorn_log_domain(
        self,
        p: np.ndarray, q: np.ndarray,
        C: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            M =  (-C + np.expand_dims(u,1) + np.expand_dims(v,0)) / self.eps
            if mask is not None:
                M[mask==0] = -1e6
            return M

        def lse(A):
            "log-sum-exp"
            max_A = np.max(A, axis=1, keepdims=True)
            return np.log(np.exp(A-max_A).sum(1, keepdims=True) + self.div_term) + max_A  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        for i in range(self.max_iters):
            u1 = u  # useful to check the update
            u = self.eps * (np.log(p) - lse(M(u, v)).squeeze()) + u
            v = self.eps * (np.log(q) - lse(M(u, v).T).squeeze()) + v
            err = np.linalg.norm(u - u1)
            if err < self.stop_thr:
                break

        U, V = u, v
        P = np.exp(M(U, V))  # P = diag(a) * K * diag(b)
        return P

    def _guide_mask(
        self,
        xs: np.ndarray, xt: np.ndarray,
        I: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        mask = np.ones((xs.shape[0], xt.shape[0]))
        mask[I, :] = 0
        mask[:, J] = 0
        mask[I, J] = 1
        return mask

    def _guide_matrix(
        self,
        Cs: np.ndarray, Ct: np.ndarray,
        I: np.ndarray, J: np.ndarray,
    ) -> np.ndarray:

        Cs_kp = Cs[:, I]
        Ct_kp = Ct[:, J]
        R1 = softmax(-2 * Cs_kp / self.rho)
        R2 = softmax(-2 * Ct_kp / self.rho)
        G = self.sim_fn(R1, R2, eps=self.div_term)
        return G

class FOT(_OT):
    def __init__(
        self,
        n_anchors: int,
        distance: Distance = SquaredEuclidean,
        sinkhorn_reg: float = 0.001, 
        stop_thr: float = 1e-7, 
        max_iters: int = 1000,
        div_term: float = 1e-10,
        sinkhorn_method: str = "sinkhorn_log"
    ):
        super().__init__(distance)
        
        self.n_anchors = n_anchors
        self.eps = sinkhorn_reg
        self.stop_thr = stop_thr
        self.max_iters = max_iters
        self.div_term = div_term
        self.sinkhorn_method = sinkhorn_method

        self.Pa_: Optional[np.ndarray] = None
        self.Pb_: Optional[np.ndarray] = None
        self.z_: Optional[np.ndarray] = None

    
    def fit(
        self, 
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        **kwargs,
    ) -> "FOT":
        z0 = self._init_anchors(xs, self.n_anchors)
        self.Pa_, self.Pb_, self.z_ = ot.factored.factored_optimal_transport(
            xs, xt, a, b,
            X0=z0,
            reg=self.eps,
            r=self.n_anchors,
            stopThr=self.stop_thr,
            numItermax=self.max_iters,
            method=self.sinkhorn_method
        )

        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        n = xs.shape[0]
        m = xt.shape[0]
        assert (self.Pa_ is not None) and (self.Pb_ is not None), "Should run fit() before mapping"
        
        Cx = self.Pa_.T.dot(xs) / (self.Pa_.T.dot(np.ones((n, 1))) + self.div_term)
        Cy = self.Pb_.dot(xt) / (self.Pb_.dot(np.ones((m, 1))) + self.div_term)
        return xs + np.dot(self.Pa_ / np.sum(self.Pa_, axis=1).reshape([n, 1]), Cy - Cx)


    def _init_anchors(
        self, 
        x: np.ndarray,
        n_clusters: int
    ) -> np.ndarray:
        model = KMeans(n_clusters=n_clusters)
        model.fit(x)
        Z = model.cluster_centers_
        return Z
    
    
# @deprecated: Adopt from https://github.com/nerdslab/latentOT
class LOT(_OT):
    def __init__(self, distance: Optional[Distance], n_source_anchors, n_target_anchors, epsilon=1, epsilon_z=1, intensity=[10, 10, 10], floyditer=50,
                 tolratio=1e-7, norm=2, random_state=None):
        super().__init__(distance)
        self.n_source_anchors, self.n_target_anchors = n_source_anchors, n_target_anchors

        self.epsilon = epsilon
        self.epsilon_z = epsilon_z

        self.intensity = intensity
        self.niter = floyditer
        self.tolratio = tolratio
        self.p = norm

        self.random_state = random_state

    @classmethod
    def compute_kmeans_centroids(cls, X, **kwargs):
        kmeans = KMeans(**kwargs).fit(X)
        return kmeans.cluster_centers_
    
    @classmethod
    def compute_cost_matrix(cls, source, target, p=2):
        cost_matrix = np.sum(np.power(source.reshape([source.shape[0], 1, source.shape[1]]) -
                                  target.reshape([1, target.shape[0], target.shape[1]]),
                                  p), axis=-1)
        return cost_matrix

    def fit(self, source: np.ndarray, target: np.ndarray, a=None, b=None, **kwargs) -> np.ndarray:
        # centroid initialized by K-means
        Cx = LOT.compute_kmeans_centroids(source, n_clusters=self.n_source_anchors, random_state=self.random_state)
        Cy = LOT.compute_kmeans_centroids(target, n_clusters=self.n_target_anchors, random_state=self.random_state)
        # Px, Py initialized by K-means and one-sided OT
        n = source.shape[0]
        m = target.shape[0]
        mu = 1 / n * np.ones([n, 1])
        nu = 1 / m * np.ones([m, 1])
        cost_xy = LOT.compute_cost_matrix(source, target, p=self.p)
        P = np.zeros([n,m]) + 1 / n / m

        converrlist = np.zeros(self.niter) + np.inf
        for t in range(0, self.niter):
            
            # compute cost matrices
            cost_x = LOT.compute_cost_matrix(source, Cx, p=self.p)
            cost_z = LOT.compute_cost_matrix(Cx, Cy, p=self.p)
            cost_y = LOT.compute_cost_matrix(Cy, target, p=self.p)
            Kx = np.exp(-self.intensity[0] * cost_x / self.epsilon)
            Kz = np.exp(-self.intensity[1] * cost_z / self.epsilon_z)
            Ky = np.exp(-self.intensity[2] * cost_y / self.epsilon)
            
            Pt1 = P
            Px, Py, Pz, P = self.update_transport_plans(Kx, Kz, Ky)  # update trans. plan

            # check for convergence
            converr = LA.norm(P - Pt1) / LA.norm(Pt1)
            converrlist[t] = converr
            if converr < self.tolratio:
                break

            # update anchors
            if t < self.niter - 1:
                Cx, Cy = self.update_anchors(Px, Py, Pz, source, target)

        self.Cx, self.Cy = Cx, Cy
        self.Px_, self.Py_, self.Pz_, self.P_ = Px, Py, Pz, P

    def update_transport_plans(self, Kx, Kz, Ky, niter=100, tol=1e-20, epsilon=0, clip_val=np.inf, epsilon1 = 0):
        dimx = Kx.shape[0]
        dimy = Ky.shape[1]
        dimz1, dimz2 = Kz.shape

        mu = 1 / dimx * np.ones([dimx, 1])
        nu = 1 / dimy * np.ones([dimy, 1])

        ax = np.ones([dimx, 1])
        bx = np.ones([dimz1, 1])
        ay = np.ones([dimz2, 1])
        by = np.ones([dimy, 1])
        az = np.ones([dimz1, 1])
        bz = np.ones([dimz2, 1])
        wxz = np.ones([dimz1, 1])
        wzy = np.ones([dimz2, 1])
        for i in range(1, niter + 1):
            
            ax = np.exp(np.minimum(np.log(np.maximum(mu,epsilon1)) - np.log(np.maximum(Kx.dot(bx), epsilon1)), clip_val))
            err1x = LA.norm(bx * Kx.T.dot(ax) - wxz, ord=1)
            

            by = np.exp(np.minimum(np.log(np.maximum(nu,epsilon1)) - np.log(np.maximum(Ky.T.dot(ay), epsilon1)), clip_val))
            err2y = LA.norm(ay * (Ky.dot(by)) - wzy, ord=1)
            
               
            wxz = ((az * (Kz.dot(bz))) * (bx * (Kx.T.dot(ax)))) ** (1 / 2)
            bx = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log( np.maximum(Kx.T.dot(ax),epsilon)), clip_val))
            err2x = LA.norm(ax * (Kx.dot(bx)) - mu, ord=1)

            az = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log(np.maximum(Kz.dot(bz), epsilon)), clip_val))
            err1z = LA.norm(bz * Kz.T.dot(az) - wzy, ord=1)
            wzy = ((ay * (Ky.dot(by))) * (bz * (Kz.T.dot(az)))) ** (1 / 2)
            bz = np.exp(np.minimum(np.log(np.maximum(wzy,epsilon)) - np.log(np.maximum(Kz.T.dot(az), epsilon)), clip_val))
            err2z = LA.norm(az * (Kz.dot(bz)) - wxz, ord=1)

            ay = np.exp(np.minimum(np.log(np.maximum(wzy, epsilon)) - np.log(np.maximum(Ky.dot(by), epsilon)), clip_val))
            err1y = LA.norm(by * Ky.T.dot(ay) - nu, ord=1)
            if max(err1x, err2x, err1z, err2z, err1y, err2y) < tol:
                break

        Px = np.diagflat(ax).dot(Kx.dot(np.diagflat(bx)))
        Pz = np.diagflat(az).dot(Kz.dot(np.diagflat(bz)))
        Py = np.diagflat(ay).dot(Ky.dot(np.diagflat(by)))
        const = 0
        z1 = Px.T.dot(np.ones([dimx, 1])) + const
        z2 = Py.dot(np.ones([dimy, 1])) + const
        P = np.dot(Px / z1.T, np.dot(Pz, Py / z2))
        return Px, Py, Pz, P

    def update_anchors(self, Px, Py, Pz, source, target):
        n = source.shape[0]
        m = target.shape[0]
        Px = self.intensity[0] * Px
        Pz = self.intensity[1] * Pz
        Py = self.intensity[2] * Py

        temp = np.concatenate((np.diagflat(Px.T.dot(np.ones([n, 1])) +
                                           Pz.dot(np.ones([self.n_target_anchors, 1]))), -Pz), axis=1)
        temp1 = np.concatenate((-Pz.T, np.diagflat(Py.dot(np.ones([m, 1])) +
                                                   Pz.T.dot(np.ones([self.n_source_anchors, 1])))), axis=1)
        temp = np.concatenate((temp, temp1), axis=0)
        sol = np.concatenate((source.T.dot(Px), target.T.dot(Py.T)), axis=1).dot(LA.inv(temp))
        Cx = sol[:, 0:self.n_source_anchors].T
        Cy = sol[:, self.n_source_anchors:self.n_source_anchors + self.n_target_anchors].T
        return Cx, Cy

    def transport(self, source, target, **kwargs) -> np.ndarray:
        n = source.shape[0]
        m = target.shape[0]
        Cx_lot = self.Px_.T.dot(source) / (self.Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
        Cy_lot = self.Py_.dot(target) / (self.Py_.dot(np.ones([m, 1])) + 10 ** -20)
        transported = source + np.dot(
            np.dot(
                self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]),
                self.Pz_ / np.sum(self.Pz_, axis=1).reshape([self.n_source_anchors, 1])
            ),
            Cy_lot) - np.dot(self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]), Cx_lot)
        return transported

    def robust_transport(self, source, target, threshold=0.8, decay=0) -> np.ndarray:
        n = source.shape[0]
        m = target.shape[0]
        Cx_lot = self.Px_.T.dot(source) / (self.Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
        Cy_lot = self.Py_.dot(target) / (self.Py_.dot(np.ones([m, 1])) + 10 ** -20)

        maxPz = np.max(self.Pz_, axis=1)
        Pz_robust = self.Pz_.copy()

        for i in range(0, self.n_source_anchors):
            for j in range(0, self.n_target_anchors):
                if self.Pz_[i, j] < maxPz[i] * threshold:
                    Pz_robust[i, j] = self.Pz_[i, j] * decay
        Pz_robust = Pz_robust / np.sum(Pz_robust, axis=1).reshape([self.n_source_anchors, 1]) * \
                    np.sum(self.Pz_, axis=1).reshape([self.n_source_anchors, 1])

        transported = source + np.dot(
            np.dot(
                self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]),
                Pz_robust / np.sum(Pz_robust, axis=1).reshape([self.n_source_anchors, 1])
            ), Cy_lot) - np.dot(self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]), Cx_lot)
        return transported