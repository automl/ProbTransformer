import pathlib
from prob_transformer.utils.handler.config import ConfigHandler
from prob_transformer.utils.logger import Logger
from prob_transformer.utils.handler.folder import FolderHandler
from prob_transformer.utils.handler.checkpoint import CheckpointHandler


class Supporter():

    def __init__(self, experiments_dir=None, config_dir=None, config_dict=None, count_expt=True):

        self.cfg = ConfigHandler(config_dir, config_dict)

        if experiments_dir is None and self.cfg.expt.experiments_dir is None:
            raise UserWarning("ConfigHandler: experiment_dir and config.expt.experiment_dir is None")
        elif experiments_dir is not None:
            self.cfg.expt.set_attr("experiments_dir", experiments_dir)
        else:
            experiments_dir = pathlib.Path(self.cfg.expt.experiments_dir)

        session_name = f"{self.cfg.data.type}-experiments"

        self.folder = FolderHandler(experiments_dir, session_name, self.cfg.expt.experiment_name, count_expt)
        self.cfg.expt.experiment_name = self.folder.experiment_name
        self.cfg.expt.experiment_dir = self.folder.dir
        self.cfg.save_config(self.folder.dir)

        self.logger = Logger(self.folder.dir)
        self.ckp = CheckpointHandler(self.folder.dir)

        self.logger.log("session_name", session_name)
        self.logger.log("experiment_name", self.cfg.expt.experiment_name)

    def get_logger(self):
        return self.logger

    def get_config(self):
        return self.cfg

    def get_checkpoint_handler(self):
        return self.ckp
