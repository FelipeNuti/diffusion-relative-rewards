import gym
import os
import pathlib
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import time
import argparse
import typing as t
from tqdm import tqdm
import plotille
import matplotlib.pyplot as plt
import imageio

WALL = 10

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, tgt, done, env_data):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())

def concat_data(data, added_data):
    N = len(added_data["observations"])
    for i in range(N):
        data['observations'].append(added_data['observations'][i])
        data['actions'].append(added_data['actions'][i])
        data['rewards'].append(0.0)
        data['terminals'].append(added_data['terminals'][i])
        data['infos/goal'].append(added_data['infos/goal'][i])
        data['infos/qpos'].append(added_data['infos/qpos'][i])
        data['infos/qvel'].append(added_data['infos/qvel'][i])

def plot2img(fig, remove_margins=True):

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

def render_field(field, title, **kwargs):
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    ax.imshow(
        field, 
        cmap=plt.cm.viridis,
        extent = (0, 1, 1, 0), 
        alpha = 1, zorder = 0, **kwargs
    ) 

    plt.axis('off')
    plt.title(title)
    img = plot2img(fig, remove_margins=False)
    plt.close()
    return img

def plot_field(trajectories, dims = (5, 7)):
    factor = 20
    dx, dy = dims
    shape = (factor * dx, factor * dy)
    bx, by = (dx, dy)
    hist_counts = np.histogram2d(
        trajectories[:, 0] + 0.5, 
        trajectories[:, 1] + 0.5, 
        bins = shape,
        range = [[0, bx], [0, by]]
    )

    counts_field = hist_counts[0] / trajectories.shape[0]
    return render_field(counts_field, title = "Counts")

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def print_spec(env):
    spec = env.str_maze_spec
    c_fill = u"\u25A0"
    positions_array = np.full(env.maze_arr.shape, 6 * c_fill)

    for x, y in env.empty_and_goal_locations:
        positions_array[x, y] = str((x, y))
    print(positions_array)

def sample_random_location(env, eps = 0.15):
    lo = np.array([0.5 + eps, 0.5 + eps])
    hi = env.dims - 1.5 - eps
    return np.random.uniform(low = lo, high = hi, size=(2,))

def is_wall(env, idx):
    w, h = idx
    return env.maze_arr[w][h] == WALL

def indices(M, N):
    for i in range(M):
        for j in range(N):
            yield np.array([i, j])

def offsets(M, N, center, diagonal = True):
    def inrange(xy):
        x, y = xy
        return x >= 0 and x < M and y >= 0 and y < N 
    for i in range(-1, 2):
        for j in range(-1, 2):
            delta = np.array([i, j])
            if (diagonal or abs(i) + abs(j) == 1) and inrange(center + delta):
                yield np.array([i, j])

def get_neighbors(M, N, k):
    """ Return values of up, down, left, and right tiles """
    neighbors = \
        [k+offset for offset in offsets(M, N, k, diagonal = True)]
    return neighbors

def dist_to_neighboring_wall(xy, env, controller):
    M, N = env.dims
    center = controller.gridify_state(xy)
    neighbors = get_neighbors(M, N, center)
    dist = 1e6
    for neigh in neighbors:
        if type(neigh) != int and is_wall(env, neigh):
            delta = neigh - center 
            mask = abs(delta) >= 1e-8
            this_dist = center + 0.5 * delta - xy
            d = np.linalg.norm(this_dist[mask])
            if d < dist:
                dist = d
    return dist

def get_sample_ranges(env, controller, dist_thresh):
    M, N = env.dims
    lims_grid = np.zeros((M, N, 2, 2))
    corner_corrections = np.zeros((M, N))
    for center in indices(M, N):
        if is_wall(env, center):
            continue
        x, y = center
        lims = np.zeros((2, 2))
        for offset in offsets(M, N, center, diagonal = True):
            if abs(offset).sum() == 2: # diagonal
                corner_point = center + offset * (0.5 - dist_thresh)
                if abs(dist_to_neighboring_wall(corner_point, env, controller) - dist_thresh*np.sqrt(2)) < 1e-4:
                    corner_corrections[x, y] += dist_thresh ** 2
            else:
                i = 1 - abs(offset[0])
                j = (1 + offset.sum()) // 2
                if is_wall(env, center + offset):
                    lims[i, j] = dist_thresh
        lims_grid[x, y, :, :] = lims
    return lims_grid, corner_corrections

def get_square_areas(lims_grid):
    widths = 1 - abs(lims_grid[:, :, 0, :]).sum(axis = -1) 
    heights = 1 - abs(lims_grid[:, :, 1, :]).sum(axis = -1)
    return widths * heights

def make_limits_and_areas_list(lims_grid, areas_grid, allowed_indices):
    lims = []
    areas = []

    for i, j in allowed_indices:
        lims.append(lims_grid[i, j])
        areas.append(areas_grid[i, j])

    return lims, np.array(areas, dtype = float)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--keep_target', type=bool, default=False, help='Keep the same target and randomize initial position when target is reached')
    parser.add_argument('--reset_when_reached', type=bool, default=False, help='Restart the environment at a random initial position when a goal is reached')
    parser.add_argument('--override_max_episode_steps', type=int, default=None, help='Override default max episode size')
    parser.add_argument('--print_spec', type = bool, default = False, help="Print the string describing the maze")
    parser.add_argument('--set_target', type = str, default=None, help="Fix a desired target in the maze, in csv (e.g. \"1,1\"). Print maze to see available positions.")
    parser.add_argument('--description', type = str, default = '', help = 'String to attach to dataset file name, e.g. maze2d-umaze-v1_\{description\}.hdf5')
    parser.add_argument('--scale_maze', type = int, default = 1, help = 'Multiply maze size')
    parser.add_argument('--openspace', type = bool, default = False, help = 'Use straight-line planner for open environments')
    parser.add_argument('--fill_square', action='store_true', help = 'Add noise to integer coordinates, such that points become uniformly sampled across space')
    parser.add_argument('--n_plot', type = int, default = 10000, help = 'Number of trajectories to plot in terminal')
    parser.add_argument('--debug', action='store_true', help='Don\'t save the dataset')
    parser.add_argument('--timestamp', action='store_true', help='Add timestamp to saved file name')
    parser.add_argument('--wall_dist_thresh', type = float, default = 0.26, help = 'Minimum distance from walls for targets and starting positions')
    parser.add_argument('--savedir', type = str, default = ".", help = "Directory in which to save the file")
    args = parser.parse_args()

    env = gym.make(args.env_name)
    maze = env.str_maze_spec
    
    if args.override_max_episode_steps is None: 
        max_episode_steps = env._max_episode_steps 
    else:
        max_episode_steps = args.override_max_episode_steps

    gain_factor = 3.0
    env = maze_model.MazeEnv(maze, scale=1)
    maze = env.str_maze_spec
    controller = waypoint_controller.WaypointController(
        maze,
        p_gain = gain_factor,
        d_gain = -1.0 * (gain_factor/10.0)**0.5,
        solve_thresh = 0.3,
        fill_space=args.fill_square,
        openspace=args.openspace,
        waypoint_noise = 0.5 * (1 - 2 * args.wall_dist_thresh)
    )

    np.set_printoptions(linewidth=10 * max(*env.dims))

    if args.print_spec:
        print_spec(env)

    global target
    target = None
    if args.set_target is not None:
        assert args.keep_target, "keep_target is false, even though a target was specified"
        target = [float(p) for p in args.set_target.split(",")]
        assert len(target) == 2, "Target position must have 2 entries"
        target = np.array(target)

    lims_grid, corner_corrections = get_sample_ranges(env, controller, args.wall_dist_thresh)
    areas_grid = get_square_areas(lims_grid)

    areas_grid -= corner_corrections

    lims, areas = make_limits_and_areas_list(lims_grid, areas_grid, env.empty_and_goal_locations)

    print(areas_grid)

    print(f"keep_target = {args.keep_target}")
    print(f"reset_when_reached = {args.reset_when_reached}")
    print(f"max_episode_steps = {max_episode_steps}")
    print(f"target: {target}")

    timestamp = int(time.time())

    def reset():
        if args.openspace:
            return env.reset_to_location(sample_random_location(env, eps = 0.26))
        else:
            s = env.reset_model(fill_square=args.fill_square, lims=lims, areas=areas)
            while (args.reset_when_reached or args.keep_target) and dist_to_neighboring_wall(s[:2], env, controller) < max(0.25, args.wall_dist_thresh):
                s = env.reset_model(fill_square=args.fill_square, lims=lims, areas=areas)
            return s

    def set_target():
        global target
        if args.openspace and not args.keep_target:
            random_target = sample_random_location(env, eps = 0.26)
            env.set_target(target_location=random_target)
            return random_target
        else:
            env.set_target(target_location=target, fill_square=args.fill_square, lims=lims, areas=areas)
            while (not args.keep_target) and dist_to_neighboring_wall(env._target, env, controller) < max(0.25, args.wall_dist_thresh):
                env.set_target(target_location=target, fill_square=args.fill_square, lims=lims, areas=areas)
            return target

    set_target()
    current_target = env._target
    s = reset()
    act = env.action_space.sample() * 1e7
    done = False

    all_data = reset_data()
    data = reset_data()
    ts = 0
    targets = []
    starting = [s[:2]]
    actions = []
    track = False
    discard = False
    progress_bar = tqdm(total = args.num_samples)
    count = 0

    while count < args.num_samples:
        position = s[0:2]
        velocity = s[2:4]

        pos_idx = controller.gridify_state(position)
        if dist_to_neighboring_wall(position, env, controller) < max(0.25, args.wall_dist_thresh): #args.wall_dist_thresh: # radius of ball is 0.2
            discard = True
        
        act, done = controller.get_action(position, velocity, current_target)
        actions.append(act)
        if ts >= max_episode_steps:
            actions = np.array(actions)
            actions_mean = abs(actions).mean(axis = 0)
            print(f"ts = {ts} | final position: {position} | final_waypoint: {controller.current_waypoint()} | distance from target: {np.linalg.norm(position - current_target)} | final idx: {controller._waypoint_idx} | avg action: {actions_mean}")
            print(f"Waypoints: {controller._waypoints}")
            print()
            done = True
        append_data(data, s, act, current_target, done, env.sim.data)

        if track and ts == 0:
            breakpoint()
        ns, _, _, _ = env.step(act)

        if ts >= max_episode_steps and controller._waypoint_idx == 0:
            discard = True

        assert (position <= env.dims).all(), "Positions out of range"

        ts += 1
        count += 1
        progress_bar.update(1)
        if discard or done:
            if not discard:
                concat_data(all_data, data)
            else:
                old_count = count
                count = max(0, count - ts)
                progress_bar.update(count - old_count)
                print(f"Run ignored | count: {old_count} -> {count}")
            data = reset_data()
            if args.keep_target:
                s = reset()
                controller._new_target(s[:2], current_target)
                env.set_target(target_location=current_target)
                act = env.action_space.sample()
            elif args.reset_when_reached:
                s = reset()
                target = set_target()
                current_target = env._target
                controller._new_target(s[:2], current_target)
            else:
                target = set_target()
                current_target = env._target
            if target is not None and np.linalg.norm(target - current_target) >= 1e-7:
                breakpoint()
            starting.append(s[:2])
            targets.append(env._target)
            actions = []
            done = False
            discard = False
            track = False
            ts = 0
        else:
            s = ns

        if args.render:
            env.render()

    progress_bar.close()
    name = args.env_name
    if args.noisy:
        name += '_noisy'
    #     name += '_single_target'
    if args.description != '':
        name += f'_{args.description}'


    targets = np.array(targets)
    starting = np.array(starting)

    obs = np.stack(all_data["observations"])
    N = min(args.num_samples, args.n_plot)

    print(f"Total of {obs.shape[0]} transitions")

    if args.debug:
        for i, lo in enumerate(range(0, args.num_samples - N, N)):
            if i > 5:
                break
            fig = plotille.Figure()
            fig.scatter(obs[lo:lo+N, 0], obs[lo:lo+N, 1], label = "Trajectory samples")
            print(fig.show(legend = True))
    else:
        fig = plotille.Figure()
        fig.scatter(obs[:N, 0], obs[:N, 1], label = "Trajectory samples")
        print(fig.show(legend = True))

    fig = plotille.Figure()
    fig.scatter(targets[:N, 0], targets[:N, 1], label = "Targets")
    print(fig.show(legend = True))

    fig = plotille.Figure()
    fig.scatter(starting[:N, 0], starting[:N, 1], label = "Starting positions")
    print(fig.show(legend = True))

    img = plot_field(obs, dims = env.dims)

    pathlib.Path(os.path.abspath(args.savedir)).mkdir(exist_ok = True, parents = True)

    if args.timestamp:
        img_name = f'heatmap_{name}_{timestamp}.png'
    else:
        img_name = f'heatmap_{name}.png'
    img_name = os.path.join(args.savedir, img_name)
    imageio.imsave(img_name, img)
  
    if args.debug:
        return
    
    if args.timestamp:
        fname = f'{name}_{timestamp}.hdf5'
    else:
        fname = f'{name}.hdf5'
    fname = os.path.join(args.savedir, fname)
    print(f"Saving to {fname}")
    dataset = h5py.File(fname, 'w')
    npify(all_data)
    for k in all_data:
        dataset.create_dataset(k, data=all_data[k], compression='gzip')


if __name__ == "__main__":
    main()
