import imageio

from src.common.utils import print_line, clear_line
from src.evaluation.base_eval import Base_Eval
from src.evaluation.utils import get_sample_trajectories


class Render(Base_Eval):
    def get_eval_name(self):
        return "render"

    def _plt_all_figs(self, save_loc: str) -> None:
        pass

    def _save_metrics(self, save_loc: str) -> None:
        self._render_env_video(save_loc)

    def _render_env_video(self, save_loc: str) -> None:

        # sample trajectory
        self.eval_env.reset()
        all_img_lists = get_sample_trajectories(
            self.eval_env, self.model, return_img=True, n_samples=1, verbose=True
        )[-1]
        all_img_lists = all_img_lists[:, 0]

        # save results
        save_file = f"{save_loc}/sample_trajectories.mp4"
        with imageio.get_writer(save_file, fps=30) as writer:
            for t, imgs in enumerate(all_img_lists):
                print_line(f"Video - Render: [{t}/{all_img_lists.shape[0]}]")
                writer.append_data(imgs)
        clear_line()
