import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gamma


class PSpline:

    def plt_axs_idx(self, iter, fig_col, row, col):
        if iter % fig_col == 0:
            r, c = row + 1, 0
        else:
            r, c = row, col + 1
        return r, c

    def t_power(self, x, knot, p):
        xx = self.newaxis(x)
        return (xx - knot) ** p * (xx > knot)

    def newaxis(self, arr):
        if isinstance(arr, np.ndarray):
            return arr[:, np.newaxis]

        return arr.to_numpy()[:, np.newaxis]

    def knots_num(self, start, stop, step):
        return int(round((stop - start) / step)) + 1

    def b_base(self, x, xl=None, xr=None, n_seg=10, b_deg=3):
        if xl is None or xl > np.min(x):
            xl = np.min(x)
            print("Left boundary adjusted to min(x) = ", xl)

        if xr is None or xr < np.max(x):
            xr = np.max(x)
            print("Right boundary adjusted to max(x) = ", xr)

        dx = (xr - xl) / n_seg
        knots_start, knot_stop = (xl - b_deg * dx), (xr + b_deg * dx)
        knots_num = self.knots_num(knots_start, knot_stop, dx)
        knots = np.linspace(knots_start, knot_stop, knots_num)
        P = self.t_power(x, knots, b_deg)
        n = P.shape[1]
        D = np.diff(np.eye(n), n=b_deg + 1) / (gamma(b_deg + 1) * dx ** b_deg)
        B = (-1) ** (b_deg + 1) * P @ D
        nb = B.shape[1]
        sk = knots[np.arange(nb) + b_deg + 1]
        Mask = self.newaxis(x) < sk
        B = B * Mask
        return B

    def hat_matrix(self, f, X):
        weights = np.diag(f.model.weights)
        cholweights = np.linalg.cholesky(weights)
        wexog = cholweights @ X
        pinv_wexog = np.linalg.pinv(wexog)
        H = cholweights @ X @ pinv_wexog
        return H

    def ps_normal(self, x, y, xl=None, xr=None, n_seg=10, b_deg=3, p_ord=2, alpha=1, wts=None, x_grid=100):
        if xl is None:
            xl = np.min(x)

        if xr is None:
            xr = np.max(x)

        m = len(x)
        B = self.b_base(x, xl=xl, xr=xr, n_seg=n_seg, b_deg=b_deg)
        n = B.shape[1]
        P = np.sqrt(alpha) * np.diff(np.eye(n), n=p_ord)
        nix = np.repeat(0, n - p_ord)
        if wts is None:
            wts = np.repeat(1, m)

        model = {}
        wtprod = np.append(wts, (nix + 1))
        X, Y = np.concatenate((B, P.T)), np.append(y, nix)
        f = sm.WLS(Y, X, weights=wtprod).fit(method='qr')
        h = self.hat_matrix(f, X).diagonal()[:m]
        beta = f.params
        mu = B @ beta
        r = (y - mu) / (1 - h)
        cv = np.sqrt(np.mean(r ** 2))
        ed = np.sum(h)
        sigma = np.sqrt(np.sum((y - mu) ** 2) / (m - ed))

        if isinstance(x_grid, int):
            x_grid = np.linspace(xl, xr, num=x_grid)
        Bu = self.b_base(x_grid, xl=xl, xr=xr, n_seg=n_seg, b_deg=b_deg)
        y_grid = Bu @ beta

        return {
            'B': B,
            'P': P,
            'mu_hat': mu,
            'cv': cv,
            'eff_dim': ed,
            'ed_resid': m - ed,
            'p_coeff': beta,
            'sigma': sigma,
            'x_grid': x_grid,
            'y_grid': y_grid
        }

    def ps_poisson(self, x, y, xl=None, xr=None, n_seg=10, b_deg=3, p_ord=2, alpha=1, wts=None, show=False,
                   iter=100, x_grid=100):
        if xl is None:
            xl = np.min(x)

        if xr is None:
            xr = np.max(x)

        m = len(x)
        B = self.b_base(x, xl=xl, xr=xr, n_seg=n_seg, b_deg=b_deg)
        n = B.shape[1]
        P = np.sqrt(alpha) * np.diff(np.eye(n), n=p_ord)
        nix = np.repeat(0, n - p_ord)
        z = np.log(y + 0.01)
        if wts is None:
            wts = np.repeat(1, m)

        model = {}
        for it in range(iter):
            mu = np.exp(z)
            w = mu
            u = (y - mu) / w + z
            wtprod = np.append(wts * w, (nix + 1))
            X, Y = np.concatenate((B, P.T)), np.append(u, nix)
            f = sm.WLS(Y, X, weights=wtprod).fit(method='qr')
            model['pcoef'] = f.params
            model['design_matrix'] = X
            znew = B @ model['pcoef']
            dz = max(abs(z - znew))
            z = znew
            if dz < 1e-06:
                break

            if show:
                print(it, dz)

        if isinstance(x_grid, int):
            x_grid = np.linspace(xl, xr, num=x_grid)
            model['x_grid'] = x_grid

        Bu = self.b_base(x_grid, xl=xl, xr=xr, n_seg=n_seg, b_deg=b_deg)
        zu = Bu @ model['pcoef']
        y_grid = zu
        model['y_grid'] = y_grid

        mu_grid = np.exp(zu)
        model['mu_grid'] = mu_grid
        return model