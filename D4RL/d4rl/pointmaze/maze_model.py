""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random


WALL = 10
EMPTY = 11
GOAL = 12

NOISE_AMOUNT = 0.5
INNER_RADIUS = 0.0


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr

def multiply_maze_size(s, scale = 1):
    assert scale % 2 == 1, "Scale must be odd"
    half_scale = (scale - 1)//2
    lines = s.split("\\")
    rows = len(lines)
    cols = len(lines[0])

    new_rows = rows * scale
    new_cols = cols * scale

    maze_line = "#" * new_cols
    s_aug = []

    for line in lines:
        line_aug = ""
        for i, c in enumerate(line):
            # if i == 0 or i == cols - 1:
            #     line_aug += c
            if c != 'G':
                line_aug += scale * c
            else:
                line_aug += half_scale * line[i-1] + c + half_scale * "O"
            
        for _ in range(half_scale):
            s_aug.append(line_aug.replace("G", "O"))
        s_aug.append(line_aug)
        for _ in range(half_scale):
            s_aug.append(line_aug.replace("G", "O"))
    
    assert len(s_aug) == new_rows, "Shape not as expected"

    print(f"Reshaped ({rows}, {cols}) -> ({new_rows}, {new_cols})")
    return "\\".join(s_aug)

def get_dims(spec):
    lines = spec.split("\\")
    return np.array([len(lines), len(lines[0])])

def point_maze(maze_str):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')
    particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
    particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    worldbody.site(name='target_site', pos=[0.0,0.0,0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])

    actuator = mjcmodel.root.actuator()
    actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
    actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    return mjcmodel


LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self,
                 maze_spec=U_MAZE,
                 reward_type='dense',
                 reset_target=False,
                 scale = 1,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.reset_target = reset_target

        self.scale = scale
        maze_spec = multiply_maze_size(maze_spec, scale = self.scale)
        self.dims = get_dims(maze_spec)

        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()

        self._target = np.array([0.0,0.0])

        model = point_maze(maze_spec)
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name, frame_skip=1)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # self.clip_velocity()
        self.do_simulation(action, self.frame_skip)
        self.set_marker()
        ob = self._get_obs()
        if self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def get_target(self):
        return self._target

    def set_target(self, target_location=None, fill_square = False, areas = None, lims = None):
        if target_location is None:
            n_positions = len(self.empty_and_goal_locations)
            if areas is None:
                areas = np.ones((n_positions, ))
            if lims is None:
                lims = np.zeros((n_positions, 2, 2))
            idx = self.np_random.choice(len(self.empty_and_goal_locations), p = (1/areas) / (1/areas).sum())
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)

            if fill_square:
                lim = lims[idx]
                noise = self.np_random.uniform(low=-0.5+lim[:, 0], high=0.5-lim[:, 1], size=self.model.nq)
                #print(f"center: {reset_location} | lims: {-0.5+lim[:, 0]} {0.5-lim[:, 1]} | noise: {noise} | pos {reset_location + noise}")
                target_location = reset_location + noise
            else:
                target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        self._target = target_location

    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id('target_site')] = np.array([self._target[0]+1, self._target[1]+1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self, fill_square = False, areas = None, lims = None):
        n_positions = len(self.empty_and_goal_locations)

        if areas is None:
            areas = np.ones((n_positions, ))
        if lims is None:
            lims = np.zeros((n_positions, 2, 2))

        idx = self.np_random.choice(len(self.empty_and_goal_locations), p = (1/areas) / (1/areas).sum())
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        if fill_square:
            #print("Fill square for position")
            lim = lims[idx]
            noise = self.np_random.uniform(low=-0.5+lim[:, 0], high=0.5-lim[:, 1], size=self.model.nq)
            # while not (abs(noise) >= NOISE_AMOUNT * INNER_RADIUS).any():
            #     noise = self.np_random.uniform(low=-NOISE_AMOUNT, high=NOISE_AMOUNT, size=self.model.nq)
            #print(noise)
            qpos = reset_location + noise
            #qpos = reset_location + self.np_random.uniform(low=0, high=2*NOISE_AMOUNT, size=self.model.nq)
        else:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = np.zeros(self.model.nv) #self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target(fill_square=fill_square)
        return self._get_obs()

    def reset_to_location(self, location, fill_square = False):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location #+ self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        if fill_square:
            noise = self.np_random.uniform(low=-NOISE_AMOUNT, high=NOISE_AMOUNT, size=self.model.nq)
            while not (abs(noise) >= NOISE_AMOUNT * INNER_RADIUS).any():
                noise = self.np_random.uniform(low=-NOISE_AMOUNT, high=NOISE_AMOUNT, size=self.model.nq)
            #noise= self.np_random.uniform(low=0, high=2*NOISE_AMOUNT, size=self.model.nq)
            #print(noise)
            qpos += reset_location + noise
        qvel = np.zeros(self.model.nv) #self.init_qvel + self.np_random.randn(self.model.nv) * .1
        #qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass

