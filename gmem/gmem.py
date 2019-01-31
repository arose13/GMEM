import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score


class GeneralMixedEffectsModel(BaseEstimator):
    def __init__(self, estimator=None, min_iterations=10, gll_early_stop_threshold=1e-3, max_iterations=20,
                 cv=3, n_jobs=1, verbose=False):
        self.verbose = verbose
        self.min_iterations = min_iterations
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        self.cv = cv

        if estimator is None:
            self.input_estimator = RandomForestRegressor(n_estimators=300, oob_score=True, n_jobs=-1)
        else:
            self.input_estimator = estimator

        self.group_counts = None
        self.estimator_ = None
        self.trained_b = None

        self.b_history = []
        self.sigma_sq_history = []
        self.D_history = []
        self.gll_history = []
        self.r2_history = []

    def predict(self, X, Z, groups):
        """
        Predict using trained General Mixed Effects Model.
        For known groups the trained random effect correction is applied.
        For unknown groups the pure fixed effect (Estimator) estimate is used.

        :param X: fixed effect covariates
        :param Z: random effect covariates
        :param groups: group assignments for samples
        :return: y, i.e. predictions
        """
        if self.estimator_ is None:
            raise NotFittedError('Model is not fitted yet. Call `fit()` first')

        if not isinstance(Z, np.ndarray):
            Z = np.array(Z)  # cast Z to numpy array (required if it's a dataframe matmul wouldn't work)

        y_pred = self.estimator_.predict(X)

        # Apply random effects correction to all known groups. Note that then, by default, the new groups get no
        # random effects correction -- which is the desired behavior.
        for group_id in self.group_counts.index:
            indices_i = groups == group_id

            # If group doesn't exist in test data that's ok. Just move on.
            if len(indices_i) == 0:
                continue

            # If group does exist, apply the correction.
            b_i = self.trained_b.loc[group_id]
            Z_i = Z[indices_i]
            y_pred[indices_i] += Z_i.dot(b_i)

        return y_pred

    def fit(self, X, Z, groups, y):
        """
        Fit GMEM using EM algorithm.

        :param X: fixed effect covariates
        :param Z: random effect covariates
        :param groups: group assignments for samples (random intercepts)
        :param y: response/target variable
        :return: fitted model
        """

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input Checks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        assert len(Z) == len(X)
        assert len(y) == len(X)
        assert len(groups) == len(X)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if isinstance(groups, np.ndarray):
            groups = pd.Series(groups.squeeze())
        elif not isinstance(groups, pd.Series):
            raise ValueError('`groups` must be an NumPy Array or Series')

        if X.ndim == 1:
            vec_shape = (len(X), 1)
            X = X.values.reshape(vec_shape) if isinstance(X, pd.Series) else X.reshape(vec_shape)

        n_groups = groups.nunique()
        n_obs = len(y)
        q = Z.shape[1]  # random effects dimension
        Z = np.array(Z)  # cast Z to numpy array (required if it's a dataframe so matmul works)

        # Create a series where group_id is the index and n_i is the value
        group_counts = groups.value_counts()

        # Do expensive slicing operations only once
        Z_by_group = {}
        y_by_group = {}
        n_by_group = {}
        I_by_group = {}
        indices_by_group = {}

        for group_id in group_counts.index:
            # Find the index for all the samples from this group in the large vector
            indices_i = groups == group_id
            indices_by_group[group_id] = indices_i

            # Slice those samples from Z and y
            Z_by_group[group_id] = Z[indices_i]
            y_by_group[group_id] = y[indices_i]

            # Get the counts for each group and create the appropriately sized identity matrix for later computations
            n_by_group[group_id] = group_counts[group_id]
            I_by_group[group_id] = np.eye(group_counts[group_id])

        # Initialize for EM algorithm
        iteration = 0
        # Note we are using a dataframe to hold the b because this is easier to index into by group_id
        # Before we were using a simple numpy array -- but we were indexing into that wrong because the group_ids
        # are not necessarily in order.
        b_df = pd.DataFrame(np.zeros((n_groups, q)), index=group_counts.index)
        sigma_sq = 1
        D = np.eye(q)

        # vectors to hold history
        self.b_history.append(b_df)
        self.sigma_sq_history.append(sigma_sq)
        self.D_history.append(D)

        while iteration < self.max_iterations:
            iteration += 1
            if self.verbose > 1:
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('Iteration: {}'.format(iteration))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ E-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # fill up y_star for all groups
            y_star = np.zeros(len(y))
            for group_id in group_counts.index:
                # Get cached group slices
                y_i = y_by_group[group_id]
                Z_i = Z_by_group[group_id]
                b_i = b_df.loc[group_id]  # used to be ix
                if self.verbose > 1:
                    print('E-step, group {}, b = {}'.format(group_id, b_i))
                indices_i = indices_by_group[group_id]

                # Compute y_star for this group and put back in right place
                y_star_i = y_i - Z_i.dot(b_i)
                y_star[indices_i] = y_star_i

            # check that still one dimensional
            # TODO: Other checks we want to do?
            assert y_star.ndim == 1

            # Do the random forest regression with all the fixed effects features
            estimator = self.input_estimator
            
            # estimated out of sample performance
            f = cross_val_predict(estimator, X, y_star, cv=self.cv, n_jobs=self.n_jobs)
            estimator.fit(X, y_star)
            oos_r2 = r2_score(y_star, f)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M-step ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            sigma_sq_sum = 0
            D_sum = 0

            for group_id in group_counts.index:
                # Get cached group slices
                indices_i = indices_by_group[group_id]
                y_i = y_by_group[group_id]
                Z_i = Z_by_group[group_id]
                n_i = n_by_group[group_id]
                I_i = I_by_group[group_id]

                # index into f
                f_i = f[indices_i]

                # Compute V_i
                V_i = Z_i.dot(D).dot(Z_i.T) + sigma_sq * I_i

                # Compute b_i
                V_inv_i = np.linalg.pinv(V_i)
                if self.verbose > 1:
                    print('M-step, pre-update, group {}, b = {}'.format(group_id, b_df.loc[group_id]))
                b_i = D.dot(Z_i.T).dot(V_inv_i).dot(y_i - f_i)
                if self.verbose > 1:
                    print('M-step, post-update, group {}, b = {}'.format(group_id, b_i))

                # Compute the total error for this group
                eps_i = y_i - f_i - Z_i.dot(b_i)

                if self.verbose > 1:
                    print('------------------------------------------')
                    print('M-step, group {}'.format(group_id))
                    print('error squared for group = {}'.format(eps_i.T.dot(eps_i)))

                # Store b for group both in numpy array and in dataframe
                # Note this HAS to be assigned with loc, otw whole df get erroneously assigned and things go to hell
                b_df.loc[group_id, :] = b_i
                if self.verbose > 1:
                    print(
                        'M-step, post-update, recalled from db, group {}, '
                        'b = {}'.format(group_id, b_df.loc[group_id])
                    )

                # Update the sums for sigma_sq and D. We will update after the entire loop over groups
                sigma_sq_sum += eps_i.T.dot(eps_i) + sigma_sq * (n_i - sigma_sq * np.trace(V_inv_i))
                D_sum += np.outer(b_i, b_i) + (
                    D - D.dot(Z_i.T).dot(V_inv_i).dot(Z_i).dot(D)
                )

            # Normalize the sums to get sigma_sq and D
            sigma_sq = (1.0 / n_obs) * sigma_sq_sum
            D = (1.0 / n_groups) * D_sum

            if self.verbose > 1:
                print('b = {}'.format(b_df))
                print('sigma_sq = {}'.format(sigma_sq))
                print('D = {}'.format(D))

            # Store off history so that we can see the evolution of the EM algorithm
            self.b_history.append(b_df.copy())
            self.sigma_sq_history.append(sigma_sq)
            self.D_history.append(D)

            # Generalized Log Likelihood computation to check convergence
            gll = 0
            for group_id in group_counts.index:
                # Get cached group slices
                indices_i = indices_by_group[group_id]
                y_i = y_by_group[group_id]
                Z_i = Z_by_group[group_id]
                I_i = I_by_group[group_id]

                # Slice f and get b
                f_i = f[indices_i]
                R_i = sigma_sq * I_i
                b_i = b_df.loc[group_id]

                # Numerically stable way of computing log(det(A))
                _, logdet_D = np.linalg.slogdet(D)
                _, logdet_R_i = np.linalg.slogdet(R_i)

                gll += (
                    (y_i - f_i - Z_i.dot(b_i))
                    .T.dot(np.linalg.pinv(R_i))
                    .dot(y_i - f_i - Z_i.dot(b_i))
                    + b_i.T.dot(np.linalg.pinv(D)).dot(b_i)
                    + logdet_D
                    + logdet_R_i
                )

            if self.verbose:
                print('R^2: {:.3f} GLL: {:.3f} at iteration {}'.format(oos_r2, gll, iteration))
            self.gll_history.append(gll)
            self.r2_history.append(oos_r2)

            # Early termination logic
            # TODO 12/22/2018 NOTE this logic assumes GLL monotonically decreases which it does not
            if len(self.gll_history) < 2:
                continue

            if abs(self.gll_history[-2] - self.gll_history[-1]) < self.gll_early_stop_threshold:
                if self.verbose:
                    print('Early Termination')
                break

        # Store off most recent model and b as the model to be used in the prediction stage
        self.group_counts = group_counts
        self.estimator_ = estimator
        self.trained_b = b_df

        return self

    def score(self, X, Z, groups, y):
        from sklearn.metrics import r2_score
        y_pred = self.predict(X, Z, groups)
        return r2_score(y, y_pred)

    def plot_history(self, figsize=(12, 4), alpha=0.5):
        import matplotlib.pyplot as graph

        graph.figure(figsize=figsize)
        graph.plot(self.r2_history)
        graph.ylabel('Out of Sample $R^2$')
        graph.xlabel('Iterations')
        graph.show()

        graph.figure(figsize=figsize)
        graph.plot(self.gll_history)
        graph.ylabel('Generalised Loglikelihood')
        graph.xlabel('Iterations')
        graph.show()

        graph.figure(figsize=figsize)
        graph.plot(np.hstack(self.b_history).T, alpha=alpha, linewidth=1)
        graph.plot(np.hstack(self.b_history).T.mean(axis=1), linestyle='--', color='black')
        graph.ylabel(r'$b$')
        graph.xlabel('Iterations')
        graph.show()

        graph.figure(figsize=figsize)
        graph.plot(self.sigma_sq_history)
        graph.ylabel(r'$\sigma^2$')
        graph.xlabel('Iterations')
        graph.show()
