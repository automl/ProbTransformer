import pathlib

from prob_transformer.utils.handler.base_handler import Handler

"""
Handle the location, new folders and experiments sub-folder structure.

base_dir / project / session / experiment

experiment will be increased

"""


class FolderHandler(Handler):

    def __init__(self, experiments_dir, session_name=None, experiment_name=None, count_expt=False,
                 reload_expt=False):
        super().__init__()

        self.experiments_dir = pathlib.Path(experiments_dir)

        self.session_name = session_name
        self.experiment_name = experiment_name
        self.count_expt = count_expt
        self.reload_expt = reload_expt

        self.expt_dir = self.create_folder()

    def create_folder(self):

        dir = self.experiments_dir
        self.save_mkdir(dir)

        dir = dir / self.session_name
        self.save_mkdir(dir)

        if self.reload_expt:
            self.experiment_name = self.get_latest_name(dir, self.experiment_name)
        elif self.count_expt:
            self.experiment_name = self.counting_name(dir, self.experiment_name)

        dir = dir / self.experiment_name
        self.save_mkdir(dir)

        return dir

    @property
    def dir(self):
        return self.expt_dir
