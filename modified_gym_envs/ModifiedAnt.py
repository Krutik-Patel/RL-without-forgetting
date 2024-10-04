from gymnasium.envs.mujoco.ant_v4 import AntEnv
import os

current_dir = os.path.dirname(os.path.realpath(__file__))

class ModifiedAntEnv(AntEnv):
    def __init__(self, render_mode=None, xml_file=os.path.join(current_dir, 'assets', 'modifiedAnt.xml')):
        super().__init__(
            xml_file=xml_file,
            render_mode=render_mode
        )

    def __str__(self):
        return 'modified_ant environment'

