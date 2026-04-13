import gymnasium as gym
from gymnasium import spaces
import mujoco
import os
import numpy as np
from PIL import Image
import random
# from .mujoco_energy import mujoco_energy

class ReachAndGraspEnv(gym.Env):
    def __init__(self, config, render_mode='human'):
        super().__init__()
        self.config = config
        
        self.objective = self.config['environment'].get('name', None)
        self.frame_skip = self.config['environment'].get('frame_skip', 1)
        self.target_threshold = self.config['environment'].get('target_threshold', 1)
        self.target_object = self.config['environment'].get('target_object')
        self.target_location_range = self.config['environment'].get('target_location_range')
        self.target_tranlation_objective = np.array(self.config['environment'].get('target_tranlation_objective', [0.0, 0.0, 0.2]))
        self.reward = self.config['environment'].get('reward', dict(weight_reach=1.0,weight_touch=1.0,weight_ori=1.0,weight_pos=1.0,weight_force=1.0))
        
        self.force_min = self.config['environment'].get('force_min', 5.0)
        self.force_max = self.config['environment'].get('force_max', 15.0)

        self.initial_joint_angle = self.config['robot'].get('initial_joint_angle',0)

        self.initial_target_quat = None
        self.initial_target_pos = None

        # Get the absolute path to the XML file
        file_name = config['robot']['name'] + '.xml'
        xml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets', file_name))
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # self.energy = mujoco_energy(self.model)

        print(f'time step {self.model.opt.timestep} sec')

        # find grasp sites in the model
        grasp_site_prefix = "site_grasp_"
        self.grasp_sites = []
        for i in range(self.model.nuser_site + self.model.nsite): # Iterate all sites
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE.value, i)
            if site_name.startswith(grasp_site_prefix):
                self.grasp_sites.append(site_name)
        self.reach_distance_min = None
        self.reach_distance_init = None
        self.is_grasped = False


        # Create a renderer
        render_width = self.config['environment'].get('render_width', 160)
        render_height = self.config['environment'].get('render_height', 120)
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.render_mode = render_mode

        # set camera
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [0,0,0]
        self.camera.distance = 0.9
        self.camera.azimuth = 90 
        self.camera.elevation = -45.0 # Look slightly down

        # Determine controllable joints
        if 'controllable_actuator' in self.config['robot'] and self.config['robot']['controllable_actuator']:
            joint_names = self.config['robot']['controllable_actuator']
            self.actuator_indices = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in joint_names]
            self.action_dim = len(self.actuator_indices)
        else:
            # If not specified, all actuators are controllable
            self.actuator_indices = list(range(self.model.nu))
            self.action_dim = self.model.nu

        self.action_range = self.model.actuator_ctrlrange[self.actuator_indices]

        # Ensure all actuator indices are valid
        assert all(i >= 0 for i in self.actuator_indices), "One or more controllable_joints not found in model actuators."

        self.save_visual_obs = self.config['environment'].get('save_visual_obs', False)
        if self.save_visual_obs:
            self.visual_obs_path = 'temp'
            os.makedirs(self.visual_obs_path, exist_ok=True)
            self.visual_obs_count = 0

        self.max_episode_steps = self.config['environment'].get('max_episode_steps', 200)
        self.step_count = 0

        self.successful_targets, self.failed_targets = [], []

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,),dtype=np.float32)

        self.observation_space = spaces.Dict()
        if self.config['environment']['feedback']['visual']:
            self.observation_space.spaces['visual'] = spaces.Box(low=0, high=255, shape=(3, render_height, render_width), dtype=np.uint8)
        if self.config['environment']['feedback']['joint']:
            self.observation_space.spaces['joint'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim*2,), dtype=np.float32)
        if self.config['environment']['feedback'].get('touch', False):
            self.observation_space.spaces['touch'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nsensor,), dtype=np.float32)

        self.render_touch = self.config['environment'].get('render_touch', False)
        print(f'render touch = {self.render_touch}')
        if self.render_touch:
            self.touch_visualisation = "on_touch"
            self.touch_color = np.array([1.0, 0.0, 0.0, 1.0])
            self.notouch_color = np.array([0.0, 1.0, 0.0, 0.2])
            
            self._touch_sensor_id_site_id = []
            for i in range(self.model.nsensor):
                if self.model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH:
                    site_id = self.model.sensor_objid[i]
                    self._touch_sensor_id_site_id.append((i, site_id))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)
        
               
        tlr = self.target_location_range
        low = np.array([tlr['x'][0], tlr['y'][0], tlr['z'][0]])
        high = np.array([tlr['x'][1], tlr['y'][1], tlr['z'][1]])
        target_pos = np.random.uniform(low=low, high=high)
        self.initial_target_pos = target_pos
        
        if self.objective == 'reach':
            self.model.body('target').pos = target_pos
            finger_tip_pos = self.data.site('finger_tip_middle').xpos
            self.reach_distance_init = np.linalg.norm(finger_tip_pos - target_pos)

        elif self.objective == 'graspf':
            
            # free target, set target position
            target_name = 'target_free_joint'
            target_joint_qpos_adr = self.model.joint('target_free_joint').qposadr
            self.data.qpos[target_joint_qpos_adr[0]:target_joint_qpos_adr[0] + 3] = target_pos

            mujoco.mj_forward(self.model, self.data) 

            grasp_site_pos = [] 
            for site_name in self.grasp_sites:
                grasp_site_pos.append(self.data.site(site_name).xpos)
            self.reach_distance_init = np.mean(np.linalg.norm(np.array(grasp_site_pos) - target_pos, axis=1))
            self.reach_distance_min = self.reach_distance_init
            
            # Store initial orientation (quaternion)
            self.initial_target_quat = self.data.qpos[target_joint_qpos_adr[0] + 3:target_joint_qpos_adr[0] + 7].copy()
            
            # Set objective position
            self.objective_pos = target_pos.copy()
            self.objective_pos += self.target_tranlation_objective
            # Objective orientation same as initial
            self.objective_quat = self.initial_target_quat.copy()

            self.is_grasped = False

        elif self.objective == 'graspj':
            
            # joystick target, set target to position
            target_base_names = ['joystick1','joystick2','joystick3']
            target_geom_names = ['joystick_knob1','joystick_knob2','joystick_knob3']
            target_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name) for target_geom_name in target_geom_names]
            
            # store original joystick position if not
            if not hasattr(self, 'initial_joystick_pos'):
                initial_data = dict()
                for target_base_name in target_base_names:
                    initial_data[target_base_name] = self.model.body(target_base_name).pos.copy()
                self.initial_joystick_pos = initial_data

            # set all joysticks position to original
            else:
                for target_base_name in target_base_names:
                    self.model.body(target_base_name).pos = self.initial_joystick_pos[target_base_name]

            # select and set target position
            n_targets = len(target_geom_ids)
            target_index = np.random.randint(0, n_targets)
            target_geom_id = target_geom_ids[target_index]
            target_base_name = target_base_names[target_index]
            target_geom_relative_pos = self.model.geom_pos[target_geom_id]
            target_pos = random.choice([low, high]) # use binary target pos
            self.model.body(target_base_name).pos = target_pos - target_geom_relative_pos
            self.target_geom_id = target_geom_id

            # Set objective position
            self.objective_pos = target_pos.copy()
            self.objective_pos += self.target_tranlation_objective

        
        # Set the robot's joint angles to a default position
        angles = np.zeros(self.model.njnt)
        if isinstance(self.initial_joint_angle, list):
            angles[0:len(self.initial_joint_angle)] = np.array(self.initial_joint_angle)
        elif isinstance(self.initial_joint_angle, dict):
            for name, value in self.initial_joint_angle.items():
                joint_obj = self.model.joint(name)
                qpos_adr = joint_obj.qposadr[0]
                angles[qpos_adr] = value
        elif isinstance(self.initial_joint_angle, (int, float)):
            angles.fill(self.initial_joint_angle)
        self.data.qpos[:self.model.njnt] = angles


        self.step_count = 0
        info = {}
        return self._get_obs(), info

    def step(self, action):
        # Initialize control vector
        ctrl = np.zeros(self.model.nu)
        # scale action to control range
        low = self.action_range[:,0]
        high = self.action_range[:,1]
        scaled_actions = (high - low) * (action + 1) / 2 + low
        # Apply the action to the robot's actuators
        ctrl[self.actuator_indices] = scaled_actions
        # self.data.ctrl[:] += ctrl # for DQN
        self.data.ctrl[:] = ctrl

        self.step_count += 1

        total_reward = 0.0
        done = False

        for _ in range(self.frame_skip):
            # Step the simulation
            mujoco.mj_step(self.model, self.data)
            
            # energy = max(self.energy.total_power(self.data), 0)
            
            # Calculate the reward
            if self.objective == 'reach':
                
                finger_tip_pos = self.data.site('finger_tip_middle').xpos
                target_pos = self.model.body('target').pos
                distance = np.linalg.norm(finger_tip_pos - target_pos)

                # total_reward += (- distance - energy*self.energy_cost_weight)/self.frame_skip
                total_reward += (-distance)/self.frame_skip
                # total_reward += np.exp(-distance)/self.frame_skip

                is_success = distance < self.target_threshold

            elif self.objective == 'graspf':
                grasp_reward, is_success = self._compute_grasp_reward_test()
                total_reward += grasp_reward / self.frame_skip
            
            elif self.objective == 'graspj':
                grasp_reward, is_success = self._compute_grasp_joystick_reward()
                total_reward += grasp_reward / self.frame_skip

            # Check if the episode is done
            done = is_success or (self.step_count >= self.max_episode_steps)
            if done:
               break

        # Get the new state of the environment
        obs = self._get_obs()
        terminated = is_success
        truncated = False
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        info = {'is_success': is_success}
        return obs, total_reward, terminated, truncated, info

    def render(self, mode='human'):
        if self.render_touch:
            self._render_callback()

        if mode == 'human':
            if not hasattr(self, 'viewer'):
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif mode == 'rgb_array':
            self.renderer.update_scene(self.data, camera=self.camera)
            return self.renderer.render()

    def _render_callback(self):
        if self.touch_visualisation == "on_touch":
            for touch_sensor_id, site_id in self._touch_sensor_id_site_id:
                # Get the address of the sensor data to handle indexing correctly
                adr = self.model.sensor_adr[touch_sensor_id]
                if self.data.sensordata[adr] > 0.01:
                    self.model.site_rgba[site_id] = self.touch_color
                else:
                    self.model.site_rgba[site_id] = self.notouch_color

    def close(self):
        if hasattr(self, 'viewer'):
            self.viewer.close()
        
    def _get_obs(self):
        obs = {}
        joint_data = []
        n = self.action_dim

        if self.config['environment']['feedback']['joint']:
            joint_data.append(self.data.qpos[:n].copy()) # joint angle
            joint_data.append(self.data.qvel[:n].copy()) # joint velocity

        if self.config['environment']['feedback'].get('touch', False):
            obs['touch'] = self.data.sensordata.copy()

        if joint_data:
            obs['joint'] = np.concatenate(joint_data)

        if self.config['environment']['feedback']['visual']:
            # Get visual feedback
            visual_obs = self.render(mode='rgb_array')
            obs['visual'] = np.moveaxis(visual_obs, 2, 0)

            if self.save_visual_obs:
                img = Image.fromarray(visual_obs)
                img.save(os.path.join(self.visual_obs_path, f'obs_{self.visual_obs_count}.png'))
                self.visual_obs_count += 1

        return obs
    
    def _compute_grasp_reward(self):

        target_pos = self.data.body('target').xpos
        target_quat = self.data.body('target').xquat

        grasp_site_pos = []
        for site_name in self.grasp_sites:
            grasp_site_pos.append(self.data.site(site_name).xpos)
        reach_distance = np.mean(np.linalg.norm(np.array(grasp_site_pos) - target_pos, axis=1))
        reach_reward = max(self.reach_distance_init - reach_distance, 0.0)
        self.reach_distance_min = min(reach_distance, self.reach_distance_min)

        target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')
        finger_geom_ids = {i for i, group in enumerate(self.model.geom_group) if group == 3}

        contacts_with_target = 0
        contact_forces = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if (geom1 == target_geom_id and geom2 in finger_geom_ids) or \
               (geom2 == target_geom_id and geom1 in finger_geom_ids):
                contacts_with_target += 1
                if contact.efc_address != -1:
                    force = self.data.efc_force[contact.efc_address]
                    contact_forces.append(force)

        # formatted_list = [f'{x:.1f}' for x in contact_forces]; print(f'{formatted_list}')

        is_touching = contacts_with_target >= 2
        touch_reward = contacts_with_target

        force_reward = 0.0
        if contact_forces:
            for force in contact_forces:
                if force < self.force_min:
                    force_reward += force
                elif force > self.force_max:
                    force_reward += -(force - self.force_max)
                else:
                    force_reward += self.force_min
            # force_reward /= len(contact_forces)

        pos_distance = np.linalg.norm(target_pos - self.objective_pos)
        pos_reward = -pos_distance

        dot_product = np.clip(np.dot(target_quat, self.objective_quat), -1.0, 1.0)
        quat_distance = 1.0 - (dot_product ** 2)
        ori_reward = -quat_distance

        # print(f"reach_reward:{reach_reward*self.reward['weight_reach']:.2f}, \
        #     touch_reward:{touch_reward*self.reward['weight_touch']:.2f}, \
        #     pos_reward:{pos_reward*self.reward['weight_pos']:.2f}, \
        #     ori_reward:{ori_reward*self.reward['weight_ori']:.2f}, \
        #     force_reward:{force_reward*self.reward.get('weight_force',0.05):.2f}")
        
        total_reward = (reach_reward * self.reward['weight_reach']
                        + touch_reward * self.reward['weight_touch']
                        + pos_reward * self.reward['weight_pos']
                        + ori_reward * self.reward['weight_ori']
                        + force_reward * self.reward.get('weight_force',0.05))

        pos_close = pos_distance < self.target_threshold
        quat_close = quat_distance < self.target_threshold
        is_success = is_touching and pos_close and quat_close

        return total_reward, is_success

    def _compute_grasp_reward_test(self):
        target_pos = self.data.body('target').xpos
        target_quat = self.data.body('target').xquat
        grasp_site_pos = []
        for site_name in self.grasp_sites:
            grasp_site_pos.append(self.data.site(site_name).xpos)

        # --- Approach Reward ---
        # r_approach = λ_approach * max(d_n* - d_n, 0)
        # d_n: current mean distance between fingertips and object
        # d_n*: stateful minimum mean distance achieved so far in episode
        reach_distance = np.mean(np.linalg.norm(np.array(grasp_site_pos) - target_pos, axis=1))
        r_approach = self.reward['weight_reach'] * max(self.reach_distance_min - reach_distance, 0.0)
        self.reach_distance_min = min(reach_distance, self.reach_distance_min)

        # --- Lift Reward ---
        # r_lift = λ_lift * max(z - z_init, 0) + I[z >= z_lifted] * B_lifted
        # z: current vertical position of object
        # z_init: initial z position of object
        # z_lifted: lifted threshold
        # B_lifted: bonus awarded at most once per episode when object has been lifted
        z = target_pos[2]
        z_init = self.initial_target_pos[2]  # set at episode reset
        z_lifted = self.objective_pos[2]

        r_lift_continuous = self.reward['weight_pos'] * max(z - z_init, 0.0)
        is_lifted = z >= z_lifted
        if is_lifted:
            lift_bonus = self.reward['lift_bonus']
        else:
            lift_bonus = 0.0
        r_lift = r_lift_continuous + lift_bonus

        # --- Grasp Reward ---
        # r_grasp = r_approach + (1 - I_grasped) * r_lift
        # I_grasped turns true once z >= z_lifted
        I_grasped = 1.0 if self.is_grasped else 0.0
        r_grasp = r_approach + (1.0 - I_grasped) * r_lift
        self.is_grasped = z >= z_lifted

        # --- Pose Rewards ---
        # pos_distance = np.linalg.norm(target_pos - self.objective_pos)
        # pos_reward = -pos_distance

        dot_product = np.clip(np.dot(target_quat, self.objective_quat), -1.0, 1.0)
        quat_distance = 1.0 - abs(dot_product)
        ori_reward = -quat_distance

        # --- Total Reward ---
        total_reward = (r_grasp
                        + ori_reward * self.reward['weight_ori'])
                        # + pos_reward * self.reward['weight_pos']

        # print(f"reach_reward:{r_approach:.2f}, \
        #     lift_reward:{r_lift:.2f}, \
        #     ori_reward:{ori_reward:.2f}, \
        # ")

        # --- Touch ---
        target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'target_geom')
        finger_geom_ids = {i for i, group in enumerate(self.model.geom_group) if group == 3}

        contacts_with_target = 0
        contact_forces = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            if (geom1 == target_geom_id and geom2 in finger_geom_ids) or \
            (geom2 == target_geom_id and geom1 in finger_geom_ids):
                contacts_with_target += 1
                if contact.efc_address != -1:
                    force = self.data.efc_force[contact.efc_address]
                    contact_forces.append(force)

        is_touching = contacts_with_target >= 2
        
        # --- Success ---
        # pos_close = pos_distance < self.target_threshold
        # quat_close = quat_distance < self.target_threshold
        # is_success = is_touching and pos_close and quat_close

        is_success = is_touching and I_grasped

        return total_reward, is_success
    

    def _compute_grasp_joystick_reward(self):

        target_pos = self.data.geom_xpos[self.target_geom_id]

        grasp_site_pos = []
        for site_name in self.grasp_sites:
            grasp_site_pos.append(self.data.site(site_name).xpos)
        reach_distance = np.mean(np.linalg.norm(np.array(grasp_site_pos) - target_pos, axis=1))
        reach_reward = -reach_distance

        # only check translation in y axis
        pos_distance = np.linalg.norm(target_pos[1] - self.objective_pos[1])
        pos_reward = -pos_distance

        # print(f"reach_reward:{reach_reward*self.reward['weight_reach']:.2f}, \
        #    pos_reward:{pos_reward*self.reward['weight_pos']:.2f}, \
        #    ")

        total_reward = (reach_reward * self.reward['weight_reach']
                        + pos_reward * self.reward['weight_pos']
                        )

        pos_close = pos_distance < self.target_threshold
        is_success = pos_close

        return total_reward, is_success