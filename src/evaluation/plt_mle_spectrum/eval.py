import os
from typing import Union, List
import json

import numpy as np

from src.common.utils import clear_line, append_to_json, print_line
from src.evaluation.base_eval import Base_Eval, Base_Collate
from src.evaluation.plt_mle_spectrum.collate import Plt_MLE_Spectrum_Collate
from src.evaluation.plt_mle_spectrum.utils import *


class Plt_MLE_Spectrum(Base_Eval):
    base_norm_period = 10
    base_n_iterations = (1000 // base_norm_period) - 1
    base_n_samples = 20
    base_eps = 1e-4
    base_lower_time = 0.0
    base_upper_time = 0.001

    @classmethod
    def get_collate_results_class(self) -> Union[Base_Collate, None]:
        return Plt_MLE_Spectrum_Collate

    @classmethod
    def get_eval_name(self):
        return "mle_spectrum"

    def _plt_all_figs(self, save_loc: str) -> None:
        all_ablation_study_function = [
            self.n_iterations_ablation_study,
            self.norm_period_ablation_study,
            self.n_samples_ablation_study,
            self.init_time_ablation_study,
            self.n_lyap_exps_ablation_study,
            self.eps_ablation_study,
        ]
        for ablation_study in all_ablation_study_function:
            ablation_study(save_loc=save_loc)

    def _save_metrics(self, save_loc: str) -> None:
        self.estimate_lyapunov_metrics(
            save_loc=save_loc,
            n_iterations=self.base_n_iterations,
            norm_period=self.base_norm_period,
            n_samples=self.base_n_samples,
            eps=self.base_eps,
            lower_time=self.base_lower_time,
            upper_time=self.base_upper_time,
        )

        self.estimate_reward_lyapunov_metrics(
            save_loc=save_loc,
            n_iterations=self.base_n_iterations,
            norm_period=self.base_norm_period,
            n_samples=self.base_n_samples,
            eps=self.base_eps,
            lower_time=self.base_lower_time,
            upper_time=self.base_upper_time,
        )

    def estimate_lyapunov_metrics(
        self,
        save_loc,
        n_iterations: int,
        norm_period: int,
        n_samples: int,
        eps: float,
        lower_time: int,
        upper_time: int,
    ):
        print("  -Estimating Lyapunov Metrics")

        # get estimated Lyapunov exponents
        lyap_exps = get_sample_estimated_lyapunov_exponents(
            env=self.eval_env,
            agent=self.model,
            n_samples=n_samples,
            n_iterations=n_iterations,
            norm_period=norm_period,
            eps=eps,
            dt=self.dt,
            lower_time=lower_time,
            upper_time=upper_time,
            verbose=True,
        )[0]

        # calculate statistics
        metrics = get_all_metrics(lyap_exps, dt=self.dt)
        save_file = f"{save_loc}/lyapunov_metrics.json"

        # save metadata
        append_to_json(
            json_file_path=save_file,
            key="info",
            val={
                "n_iterations": n_iterations,
                "norm_period": norm_period,
                "n_samples": n_samples,
                "eps": eps,
                "n_model_steps": self.model.num_timesteps,
                "dt": self.dt,
            },
        )

        # save metric results
        append_to_json(
            json_file_path=save_file,
            key="metrics",
            val={
                key: get_series_statistics(val, return_tuple=False)
                for (key, val) in metrics.items()
            },
        )

        # save Lyapunov exponents
        lyap_exps_stats = get_series_statistics(lyap_exps, return_tuple=False)
        lyap_exps_stats["mean"] = lyap_exps_stats["mean"].tolist()
        lyap_exps_stats["std"] = lyap_exps_stats["std"].tolist()
        conf_interval = lyap_exps_stats.pop("ci")
        lyap_exps_stats["lower_ci"] = conf_interval[0].tolist()
        lyap_exps_stats["upper_ci"] = conf_interval[1].tolist()

        append_to_json(
            json_file_path=save_file,
            key="Lyapunov exponents",
            val=lyap_exps_stats,
        )

        append_to_json(
            json_file_path=save_file,
            key="raw",
            val=lyap_exps.tolist(),
        )

    def estimate_reward_lyapunov_metrics(
        self,
        save_loc,
        n_iterations: int,
        norm_period: int,
        n_samples: int,
        eps: float,
        lower_time: int,
        upper_time: int,
    ):
        print("  -Estimating Reward Lyapunov Metrics")

        # get estimated Lyapunov exponents
        reward_lyap_exps = get_sample_estimated_lyapunov_exponents(
            env=self.eval_env,
            agent=self.model,
            n_samples=n_samples,
            n_iterations=n_iterations,
            norm_period=norm_period,
            eps=eps,
            dt=self.dt,
            lower_time=lower_time,
            upper_time=upper_time,
            verbose=True,
            n_lyap_exps=1,
        )[2]

        # # calculate statistics
        # metrics = get_all_metrics(lyap_exps, dt=self.dt)
        save_file = f"{save_loc}/reward_lyapunov_metrics.json"

        # save metadata
        append_to_json(
            json_file_path=save_file,
            key="info",
            val={
                "n_iterations": n_iterations,
                "norm_period": norm_period,
                "n_samples": n_samples,
                "eps": eps,
                "n_model_steps": self.model.num_timesteps,
                # "n_env_steps": self.eval_env.obs_rms.count,
            },
        )

        mle_metrics = get_maximum_lyap_exponent(reward_lyap_exps)
        mle_metric_stats = get_series_statistics(mle_metrics, return_tuple=False)

        # # save metric results
        append_to_json(
            json_file_path=save_file,
            key="metrics",
            val={"reward_MLE": mle_metric_stats},
        )

        append_to_json(
            json_file_path=save_file,
            key="raw",
            val=reward_lyap_exps.tolist(),
        )

    def n_iterations_ablation_study(
        self,
        save_loc: str,
        min_iterations: int = 100,
        max_iterations: int = 1000,
        norm_period: int = base_norm_period,
        n_samples: int = base_n_samples,
        eps: float = base_eps,
        lower_time: int = base_lower_time,
        upper_time: int = base_upper_time,
    ) -> None:
        print("  -Running n_iterations ablation study")

        # get Lyapunov exponent history
        os.makedirs(f"{save_loc}/n_iterations", exist_ok=True)
        lyap_values_save_loc = f"{save_loc}/n_iterations/values.json"

        if os.path.isfile(lyap_values_save_loc):
            with open(lyap_values_save_loc, "r") as json_file:
                lyap_exps_hist = json.load(json_file)
                lyap_exps_hist = np.array(lyap_exps_hist)
        else:
            lyap_exps_hist = get_sample_estimated_lyapunov_exponents(
                env=self.eval_env,
                agent=self.model,
                n_samples=n_samples,
                n_iterations=max_iterations,
                norm_period=norm_period,
                eps=eps,
                dt=self.dt,
                lower_time=lower_time,
                upper_time=upper_time,
                verbose=True,
            )[1]

            with open(lyap_values_save_loc, "w+") as json_file:
                json.dump(lyap_exps_hist.tolist(), json_file, indent=4)

        x_values = np.arange(min_iterations, max_iterations)
        lyap_exps_hist = lyap_exps_hist[min_iterations:]

        # plot results
        plot_lyapunov_exponents_and_metrics(
            lyap_exps_list=lyap_exps_hist,
            save_loc=save_loc,
            study_name="n_iterations",
            dt=self.dt,
            x_values=x_values,
            x_label="Number of iterations",
            exponent_title="Convergence of Lyapunov Exponents",
            metric_title="Lyapunov metrics for increasing number of interactions",
        )

    def eps_ablation_study(
        self,
        save_loc: str,
        n_iterations: int = base_n_iterations,
        norm_period: int = base_norm_period,
        n_samples: int = base_n_samples,
        eps_values: List[float] = [1e-5, 1e-4, 1e-3],
        lower_time: int = base_lower_time,
        upper_time: int = base_upper_time,
    ) -> None:
        print("  -Running eps ablation study")

        # get sample lyapunov exponents
        all_lyap_exps = []
        for eps in eps_values:
            print_line(f"Eps = {eps}")
            lyap_exps = get_sample_estimated_lyapunov_exponents(
                env=self.eval_env,
                agent=self.model,
                n_samples=n_samples,
                n_iterations=n_iterations,
                norm_period=norm_period,
                eps=eps,
                dt=self.dt,
                lower_time=lower_time,
                upper_time=upper_time,
            )[0]
            all_lyap_exps.append(lyap_exps)
        all_lyap_exps = np.array(all_lyap_exps)
        clear_line()

        # plot results
        plot_lyapunov_exponents_and_metrics(
            lyap_exps_list=all_lyap_exps,
            save_loc=save_loc,
            study_name="eps",
            dt=self.dt,
            x_values=eps_values,
            x_label="Perturbation size",
            x_scale="log",
            exponent_title="Affects of perturbation size on Lyapunov exponents",
            metric_title="Lyapunov metrics for values of epsilon",
        )

    def norm_period_ablation_study(
        self,
        save_loc: str,
        n_iterations: int = base_n_iterations,
        norm_period_values: List[int] = [1, 10, 100],
        n_samples: int = base_n_samples,
        eps: float = base_eps,
        lower_time: int = base_lower_time,
        upper_time: int = base_upper_time,
    ) -> None:
        print("  -Running norm period ablation study")

        # get Lyapunov exponent history
        os.makedirs(f"{save_loc}/norm_period", exist_ok=True)
        lyap_values_save_loc = f"{save_loc}/norm_period/values.json"

        if os.path.isfile(lyap_values_save_loc):
            with open(lyap_values_save_loc, "r") as json_file:
                all_lyap_exps = json.load(json_file)
                all_lyap_exps = np.array(all_lyap_exps)
        else:
            # get sample lyapunov exponents
            all_lyap_exps = []
            for norm_period in norm_period_values:
                print_line(f"Norm Period = {norm_period}")
                lyap_exps = get_sample_estimated_lyapunov_exponents(
                    env=self.eval_env,
                    agent=self.model,
                    n_samples=n_samples,
                    n_iterations=n_iterations,
                    norm_period=norm_period,
                    eps=eps,
                    dt=self.dt,
                    lower_time=lower_time,
                    upper_time=upper_time,
                    verbose=True,
                )[0]
                all_lyap_exps.append(lyap_exps)
            all_lyap_exps = np.array(all_lyap_exps)

            with open(lyap_values_save_loc, "w+") as json_file:
                json.dump(all_lyap_exps.tolist(), json_file, indent=4)

        clear_line()

        # plot results
        plot_lyapunov_exponents_and_metrics(
            lyap_exps_list=all_lyap_exps,
            save_loc=save_loc,
            study_name="norm_period",
            dt=self.dt,
            x_values=norm_period_values,
            x_scale="log",
            x_label="Normalization Period",
            exponent_title="Affects of normalization period on Lyapunov exponents",
            metric_title="Lyapunov metrics for normalization periods",
        )

    def n_samples_ablation_study(
        self,
        save_loc: str,
        n_iterations: int = base_n_iterations,
        norm_period: int = base_norm_period,
        max_n_samples: int = 20,
        eps: float = base_eps,
        lower_time: int = base_lower_time,
        upper_time: int = base_upper_time,
    ) -> None:
        print("  -Running number of samples ablation study")

        # get Lyapunov exponent history
        os.makedirs(f"{save_loc}/n_samples", exist_ok=True)
        lyap_values_save_loc = f"{save_loc}/n_samples/values.json"

        if os.path.isfile(lyap_values_save_loc):
            with open(lyap_values_save_loc, "r") as json_file:
                lyap_exps = json.load(json_file)
                lyap_exps = np.array(lyap_exps)
        else:
            lyap_exps = get_sample_estimated_lyapunov_exponents(
                env=self.eval_env,
                agent=self.model,
                n_samples=max_n_samples,
                n_iterations=n_iterations,
                norm_period=norm_period,
                eps=eps,
                dt=self.dt,
                lower_time=lower_time,
                upper_time=upper_time,
                verbose=True,
            )[0]

            with open(lyap_values_save_loc, "w+") as json_file:
                json.dump(lyap_exps.tolist(), json_file, indent=4)

        lyap_exps_list = [lyap_exps[: i + 1] for i in range(max_n_samples)]

        # plot results
        plot_lyapunov_exponents_and_metrics(
            lyap_exps_list=lyap_exps_list,
            save_loc=save_loc,
            study_name="n_samples",
            dt=self.dt,
            x_values=np.arange(max_n_samples) + 1,
            x_label="Number of samples",
            exponent_title="Affects of using more samples on Lyapunov exponents",
            metric_title="Lyapunov metrics for number of samples used",
        )

    def init_time_ablation_study(
        self,
        save_loc: str,
        n_iterations: int = base_n_iterations,
        norm_period: int = base_norm_period,
        n_samples: int = base_n_samples,
        eps: float = base_eps,
        lower_time: int = 200,
        upper_time: int = 800,
        time_period: int = 200,
    ):
        print("  -Running time period ablation study")

        all_lyap_exps = []
        for time_interval in range(lower_time, upper_time, time_period):
            lyap_exps = get_sample_estimated_lyapunov_exponents(
                env=self.eval_env,
                agent=self.model,
                n_samples=n_samples,
                n_iterations=n_iterations,
                norm_period=norm_period,
                eps=eps,
                dt=self.dt,
                lower_time=time_interval,
                upper_time=time_interval + time_period,
            )[0]
            all_lyap_exps.append(lyap_exps)
        all_lyap_exps = np.array(all_lyap_exps)

        # plot results
        plot_lyapunov_exponents_and_metrics(
            lyap_exps_list=all_lyap_exps,
            save_loc=save_loc,
            study_name="time_period",
            dt=self.dt,
            x_values=np.arange(lower_time, upper_time, time_period),
            x_label="Start Time",
            exponent_title="Affects of different initial sample times on Lyapunov exponents",
            metric_title="Lyapunov metrics for sample time used",
        )

    def n_lyap_exps_ablation_study(
        self,
        save_loc: str,
        n_iterations: int = base_n_iterations,
        norm_period: int = base_norm_period,
        n_samples: int = base_n_samples,
        eps: float = base_eps,
        lower_time: int = base_lower_time,
        upper_time: int = base_upper_time,
    ):
        print("  -Running #Lyapunov exponents ablation study")
        max_n_dims = self.eval_env.env_method("get_state")[0].shape[-1]

        all_lyap_exps = []
        for n_dims in range(1, max_n_dims + 1):
            print_line(f"{n_dims}/{max_n_dims}")
            lyap_exps = get_sample_estimated_lyapunov_exponents(
                env=self.eval_env,
                agent=self.model,
                n_samples=n_samples,
                n_iterations=n_iterations,
                norm_period=norm_period,
                eps=eps,
                dt=self.dt,
                lower_time=lower_time,
                upper_time=upper_time,
                n_lyap_exps=n_dims,
            )[0]
            all_lyap_exps.append(lyap_exps)
        clear_line()

        # plot results
        plot_lyapunov_exponents_and_metrics(
            lyap_exps_list=all_lyap_exps,
            save_loc=save_loc,
            study_name="n_lyap_exps",
            dt=self.dt,
            x_values=np.arange(1, max_n_dims + 1),
            x_label="#Lyapunov Exponents",
            exponent_title="Affects of different number of Lyapunov exponents on convergence",
            metric_title="Lyapunov metrics for varying number of Lyapunov Exponents",
        )
