

class JournalAdjustment(JournalBase):
    pass


###
###
###
###
class JournalBase:
    """Journal.

    """
    pass
    '''
    def __init__(self):

        self.accepted_parameters = {}
        self.parameter_names = []
        self.parameter_names_tex = []
        self.labels = []
        self.distances = []
        self.sumstats = []
        self._n_parameters = 0

        self.configuration = {}
        self.sampler_summary = {}

        self._journal_started = False

    def _start_journal(self):
        self._journal_started = True

    def _add_config(self, simulator, inference_scheme, distance, n_simulations, epsilon):
        self.configuration["Simulator model"] = simulator.__name__
        self.configuration["Inference scheme"] = inference_scheme
        self.configuration["Distance metric"] = distance.__name__
        self.configuration["Number of simulations"] = n_simulations
        self.configuration["Epsilon"] = epsilon

    def _add_parameter_names(self, priors):
        for parameter in priors:
            name = parameter.name
            tex = parameter.tex
            self.parameter_names.append(name)
            self.accepted_parameters[name] = []
            self.parameter_names_tex.append(tex)
            self._n_parameters += 1
            if tex is None:
                self.labels.append(name)
            else:
                self.labels.append(tex)

    def _add_accepted_parameters(self, thetas):
        for parameter_name, theta in zip(self.parameter_names, thetas):
            self.accepted_parameters[parameter_name].append(theta)

    def _add_distance(self, distance):
        self.distances.append(distance)

    def _add_sumstat(self, sumstat):
        self.sumstats.append(sumstat)

    def _add_sampler_summary(self, number_of_simulations, accepted_count):
        accept_ratio = accepted_count / number_of_simulations
        # number of parameters estimated
        self.sampler_summary["Number of simulations"] = number_of_simulations
        self.sampler_summary["Number of accepted simulations"] = accepted_count
        self.sampler_summary["Acceptance ratio"] = accept_ratio
        # posterior means
        # uncertainty

    @property
    def get_accepted_parameters(self):
        return self.accepted_parameters

    def _get_params_as_arrays(self):
        """
        Transform data of accepted parameters to 1D arrays
        """

        samples = self.get_accepted_parameters
        if len(self.parameter_names) > 1:
            params = (np.asarray(samples[name], float).squeeze() if np.asarray(
                samples[name], float).ndim > 1 else np.asarray(samples[name], float) for name in self.parameter_names)
        else:
            samples = np.asarray(samples[self.parameter_names[0]], float)
            params = samples.squeeze() if samples.ndim > 1 else samples
        return params

    def params_as_arrays(self):
        *data, = self._get_params_as_arrays()
        return data

    def _set_point_estimate_statistic(self, statistic):
        if statistic == 'mean':
            pass

    def _sample_point_estimates(self, statistic):
        """
        Calculate point estimate of inferred parameters.

        In statistics, point estimation involves the use of sample data to
        calculate a single value (known as a point estimate since it identifies
        a point in some parameter space) which is to serve as a "best guess" or
        "best estimate" of an unknown population parameter (for example,
        the population mean).

        https://en.wikipedia.org/wiki/Point_estimation
        """
        *samples, = self._get_params_as_arrays()
        if self._n_parameters == 1:
            point_estimates = [np.mean(samples)]
        else:
            point_estimates = [np.mean(sample) for sample in samples]
        return point_estimates

    @property
    def get_distances(self):
        check_journal_status(self._journal_started)
        return self.distances

    @property
    def get_number_of_simulations(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Number of simulations"]

    @property
    def get_number_of_accepted_simulations(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Number of accepted simulations"]

    @property
    def get_acceptance_ratio(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Acceptance ratio"]

    def _samples(self, name):
        pass

    def _kde(self):
        pass

    def _add_histplot(self, data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values):
        n_bins = self._freedman_diaconis_rule(data)
        ax.hist(data, density=density, histtype='bar', edgecolor=None,
                color='steelblue', alpha=0.5, bins=n_bins, label="Accepted samples")
        ax.axvline(
            point_estimates[index], color='b', label="Point estimate")
        if true_vals_bool:
            ax.axvline(
                true_parameter_values[index], color='r', linestyle='--', label="Groundtruth")
        ax.set_xlabel(self.labels[index])
        ax.set_title("Histogram of accepted " + self.labels[index])


    def histplot(self, bins=10, rug=False, point_estimate='mean', show=True, dpi=120, path_to_save=None, true_parameter_values=None, **kwargs):
        """
        histogram(s) of sampled parameter(s)

        point estimate : mean, median, mode, None

        The Mode value is the value that appears the most number of times
        The median value is the value in the middle, after you have sorted all the values
        The mean value is the average value
        """

        N = self._n_parameters
        # run checks
        check_journal_status(self._journal_started)

        if point_estimate is not None:
            check_point_estimate_input(point_estimate)
            point_estimates = self._point_estimates()

        true_vals_bool = False
        if true_parameter_values is not None:
            check_true_parameter_values(N, true_parameter_values)
            true_vals_bool = True

        # get sampled parameters
        *data, = self._get_params_as_arrays()

        fig = plt.figure(figsize=(8, 6), tight_layout=True, dpi=dpi)
        self._set_plot_style()

        if N == 1:
            ax = plt.subplot(111)
            legend_position = 0
            index = 0
            add_histplot(data, ax, bins, label="Accepted samples", **kwargs)
            if rug:
                add_rugplot(data, ax)
            if decorate:
                ax.axvline(point_estimates[index],
                           color='b', label="Point estimate")

            self._add_histplot(
                data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)
        else:
            if N == 2 or N == 4:
                cols = 2
                legend_position = 1
            else:
                cols = 3
                legend_position = 2
            rows = int(np.ceil(N / cols))
            gs = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)
            for index, data in enumerate(data):
                ax = fig.add_subplot(gs[index])
                self._add_histplot(
                    data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)

        handles, labels = plt.gca().get_legend_handles_labels()
        if true_vals_bool:
            order = [2, 0, 1]
        else:
            order = [1, 0]

        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='center left',
                   bbox_to_anchor=(1.04, 0.5),
                   fancybox=True,
                   borderaxespad=0.1,
                   ncol=1
                   )

        if path_to_save is not None:
            fig.savefig(path_to_save, dpi=dpi)
        if show:
            plt.show()

    def adjusted_histplot():
        # regression adjusted
        pass

    def kdeplot(self, kernel="gaussian"):
        ax[1].plot(x, kernel.evaluate(x), label="approximate posterior")
        pass

    def distplot(self, kde=True, kde_kwds=None, ax=None):
        """
        """
        if ax is None:
            ax = plt.gca()
        pass

    def posterior_kde(self, kernel="gaussian"):
        pass

    @ property
    def summary(self):
        pass

    @ property
    def print_summary(self):
        pass

    def save(self, filename):
        """
        Stores the journal to disk.

        Parameters
        ----------
        filename: string
            the location of the file to store the current object to.
        """

        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)

    def load(self, filename):
        with open(filename, 'rb') as input:
            journal = pickle.load(input)
        return journal

    def kde_fit(self):
        # move to _adjustment
        # wrap kde class instance
        # to be called from posterior_plot
        pass

    def run_lra(self):
        # move to _adjustment
        # is_performed = True # when already run, do not run again
        pass

    '''

    '''
    @staticmethod
    def run_lra(
        theta: torch.Tensor,
        x: torch.Tensor,
        observation: torch.Tensor,
        sample_weight=None,
    ) -> torch.Tensor:
        """Return parameters adjusted with linear regression adjustment.
        Implementation as in Beaumont et al. 2002: https://arxiv.org/abs/1707.01254
        """

        theta_adjusted = theta
        for parameter_idx in range(theta.shape[1]):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=x,
                y=theta[:, parameter_idx],
                sample_weight=sample_weight,
            )
            theta_adjusted[:, parameter_idx] += regression_model.predict(
                observation.reshape(1, -1)
            )
            theta_adjusted[:, parameter_idx] -= regression_model.predict(x)

        return theta_adjusted
    '''

##
###
##
##
##


'''
class Journal:
    """Journal.

    """

    def __init__(self):

        self.accepted_parameters = {}
        self.parameter_names = []
        self.parameter_names_tex = []
        self.labels = []
        self.distances = []
        self.sumstats = []
        self._n_parameters = 0

        self.configuration = {}
        self.sampler_summary = {}

        self._journal_started = False

    def _start_journal(self):
        self._journal_started = True

    def _add_config(self, simulator, inference_scheme, distance, n_simulations, epsilon):
        self.configuration["Simulator model"] = simulator.__name__
        self.configuration["Inference scheme"] = inference_scheme
        self.configuration["Distance metric"] = distance.__name__
        self.configuration["Number of simulations"] = n_simulations
        self.configuration["Epsilon"] = epsilon

    def _add_parameter_names(self, priors):
        for parameter in priors:
            name = parameter.name
            tex = parameter.tex
            self.parameter_names.append(name)
            self.accepted_parameters[name] = []
            self.parameter_names_tex.append(tex)
            self._n_parameters += 1
            if tex is None:
                self.labels.append(name)
            else:
                self.labels.append(tex)

    def _add_accepted_parameters(self, thetas):
        for parameter_name, theta in zip(self.parameter_names, thetas):
            self.accepted_parameters[parameter_name].append(theta)

    def _add_distance(self, distance):
        self.distances.append(distance)

    def _add_sumstat(self, sumstat):
        self.sumstats.append(sumstat)

    def _add_sampler_summary(self, number_of_simulations, accepted_count):
        accept_ratio = accepted_count / number_of_simulations
        # number of parameters estimated
        self.sampler_summary["Number of simulations"] = number_of_simulations
        self.sampler_summary["Number of accepted simulations"] = accepted_count
        self.sampler_summary["Acceptance ratio"] = accept_ratio
        # posterior means
        # uncertainty

    @property
    def get_accepted_parameters(self):
        return self.accepted_parameters

    def _get_params_as_arrays(self):
        """
        Transform data of accepted parameters to 1D arrays
        """

        samples = self.get_accepted_parameters
        if len(self.parameter_names) > 1:
            params = (np.asarray(samples[name], float).squeeze() if np.asarray(
                samples[name], float).ndim > 1 else np.asarray(samples[name], float) for name in self.parameter_names)
        else:
            samples = np.asarray(samples[self.parameter_names[0]], float)
            params = samples.squeeze() if samples.ndim > 1 else samples
        return params

    def _set_point_estimate_statistic(self, statistic):
        if statistic == 'mean':
            pass

    def _sample_point_estimates(self, statistic):
        """
        Calculate point estimate of inferred parameters.

        In statistics, point estimation involves the use of sample data to
        calculate a single value (known as a point estimate since it identifies
        a point in some parameter space) which is to serve as a "best guess" or
        "best estimate" of an unknown population parameter (for example,
        the population mean).

        https://en.wikipedia.org/wiki/Point_estimation
        """
        *samples, = self._get_params_as_arrays()
        if self._n_parameters == 1:
            point_estimates = [np.mean(samples)]
        else:
            point_estimates = [np.mean(sample) for sample in samples]
        return point_estimates

    @property
    def get_distances(self):
        check_journal_status(self._journal_started)
        return self.distances

    @property
    def get_number_of_simulations(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Number of simulations"]

    @property
    def get_number_of_accepted_simulations(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Number of accepted simulations"]

    @property
    def get_acceptance_ratio(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Acceptance ratio"]

    def _samples(self, name):
        pass

    def _kde(self):
        pass
'''
'''
    def _add_histplot(self, data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values):
        n_bins = self._freedman_diaconis_rule(data)
        ax.hist(data, density=density, histtype='bar', edgecolor=None,
                color='steelblue', alpha=0.5, bins=n_bins, label="Accepted samples")
        ax.axvline(
            point_estimates[index], color='b', label="Point estimate")
        if true_vals_bool:
            ax.axvline(
                true_parameter_values[index], color='r', linestyle='--', label="Groundtruth")
        ax.set_xlabel(self.labels[index])
        ax.set_title("Histogram of accepted " + self.labels[index])
    '''
'''
    def histplot(self, bins=10, rug=False, point_estimate='mean', show=True, dpi=120, path_to_save=None, true_parameter_values=None, **kwargs):
        """
        histogram(s) of sampled parameter(s)

        point estimate : mean, median, mode, None

        The Mode value is the value that appears the most number of times
        The median value is the value in the middle, after you have sorted all the values
        The mean value is the average value
        """

        N = self._n_parameters
        # run checks
        check_journal_status(self._journal_started)

        if point_estimate is not None:
            check_point_estimate_input(point_estimate)
            point_estimates = self._point_estimates()

        true_vals_bool = False
        if true_parameter_values is not None:
            check_true_parameter_values(N, true_parameter_values)
            true_vals_bool = True

        # get sampled parameters
        *data, = self._get_params_as_arrays()

        fig = plt.figure(figsize=(8, 6), tight_layout=True, dpi=dpi)
        self._set_plot_style()

        if N == 1:
            ax = plt.subplot(111)
            legend_position = 0
            index = 0
            add_histplot(data, ax, bins, label="Accepted samples", **kwargs)
            if rug:
                add_rugplot(data, ax)
            if decorate:
                ax.axvline(point_estimates[index],
                           color='b', label="Point estimate")

            self._add_histplot(
                data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)
        else:
            if N == 2 or N == 4:
                cols = 2
                legend_position = 1
            else:
                cols = 3
                legend_position = 2
            rows = int(np.ceil(N / cols))
            gs = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)
            for index, data in enumerate(data):
                ax = fig.add_subplot(gs[index])
                self._add_histplot(
                    data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)

        handles, labels = plt.gca().get_legend_handles_labels()
        if true_vals_bool:
            order = [2, 0, 1]
        else:
            order = [1, 0]

        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='center left',
                   bbox_to_anchor=(1.04, 0.5),
                   fancybox=True,
                   borderaxespad=0.1,
                   ncol=1
                   )

        if path_to_save is not None:
            fig.savefig(path_to_save, dpi=dpi)
        if show:
            plt.show()

    def adjusted_histplot():
        # regression adjusted
        pass

    def kdeplot(self, kernel="gaussian"):
        ax[1].plot(x, kernel.evaluate(x), label="approximate posterior")
        pass

    def distplot(self, kde=True, kde_kwds=None, ax=None):
        """
        """
        if ax is None:
            ax = plt.gca()
        pass

    def posterior_kde(self, kernel="gaussian"):
        pass

    @ property
    def summary(self):
        pass

    @ property
    def print_summary(self):
        pass

    def save(self, filename):
        """
        Stores the journal to disk.

        Parameters
        ----------
        filename: string
            the location of the file to store the current object to.
        """

        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)

    def load(self, filename):
        with open(filename, 'rb') as input:
            journal = pickle.load(input)
        return journal

    def kde_fit(self):
        # move to _adjustment
        # wrap kde class instance
        # to be called from posterior_plot
        pass

    def run_lra(self):
        # move to _adjustment
        # is_performed = True # when already run, do not run again
        pass
'''

'''
    @staticmethod
    def run_lra(
        theta: torch.Tensor,
        x: torch.Tensor,
        observation: torch.Tensor,
        sample_weight=None,
    ) -> torch.Tensor:
        """Return parameters adjusted with linear regression adjustment.
        Implementation as in Beaumont et al. 2002: https://arxiv.org/abs/1707.01254
        """

        theta_adjusted = theta
        for parameter_idx in range(theta.shape[1]):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=x,
                y=theta[:, parameter_idx],
                sample_weight=sample_weight,
            )
            theta_adjusted[:, parameter_idx] += regression_model.predict(
                observation.reshape(1, -1)
            )
            theta_adjusted[:, parameter_idx] -= regression_model.predict(x)

        return theta_adjusted
    '''
