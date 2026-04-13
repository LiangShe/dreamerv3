import embodied
import elements
import numpy as np
import yaml
import os

from . import reach_grasp_env_gym

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class armctrl(embodied.Env):

  def __init__(self, task='graspj', size=(128, 128), image=True, proprio=True, touch=True, **kwargs):
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, 'config_' + task + '.yaml')
    self._config = load_config(config_path)

    if image:
        self._config['environment']['feedback']['visual'] = True
    if proprio:
        self._config['environment']['feedback']['joint'] = True
    if touch:
        self._config['environment']['feedback']['touch'] = True
    self._config['environment']['render_width'] = size[0]
    self._config['environment']['render_height'] = size[1]
    self._env = reach_grasp_env_gym.ReachAndGraspEnv(self._config, render_mode='rgb_array')

    self._size = size
    self._image = image
    self._proprio = proprio
    self._touch = touch

    self._done = True

  @property
  def obs_space(self):
    spaces = {
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self._image:
      spaces['image'] = elements.Space(np.uint8, self._size + (3,))
    if self._proprio:
      spaces['joint'] = self._env.observation_space.spaces['joint']
    if self._touch:
      spaces['touch'] = self._env.observation_space.spaces['touch']

    return spaces

  @property
  def act_space(self):
    spec = self._env.action_space
    return {
        'action': elements.Space(spec.dtype, spec.shape, spec.low, spec.high),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, info = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)

    obs, reward, terminated, truncated, info = self._env.step(action['action'])
    self._done = terminated or truncated
    return self._obs(obs, reward, is_last=self._done, is_terminal=terminated)

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    obs['image'] = np.moveaxis(obs.pop('visual'), 0, 2)
    obs['reward'] = np.float32(reward)
    obs['is_first'] = is_first
    obs['is_last'] = is_last
    obs['is_terminal'] = is_terminal
    return obs