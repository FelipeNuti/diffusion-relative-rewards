import numpy as np
from d4rl.pointmaze import q_iteration
from d4rl.pointmaze.maze_model import multiply_maze_size
from d4rl.pointmaze.gridcraft import grid_env
from d4rl.pointmaze.gridcraft import grid_spec


ZEROS = np.zeros((2,), dtype=np.float32)
ONES = np.zeros((2,), dtype=np.float32)


class WaypointController(object):
    def __init__(self, maze_str, openspace = False, fill_space = False, scale = 1, waypoint_noise = 0, solve_thresh=0.1, p_gain=10.0, d_gain=-1.0):
        # self.original_maze_str = maze_str
        self.maze_str = maze_str
        self.openspace = openspace
        self.fill_space = fill_space
        self._target = -1000 * ONES
        self.waypoint_noise = waypoint_noise

        self.p_gain = p_gain * scale
        self.d_gain = d_gain * scale
        self.solve_thresh = solve_thresh * scale #* 10.0/p_gain
        self.vel_thresh = 0.1 * scale

        self._waypoint_idx = 0
        self._waypoints = []
        self._waypoint_prev_loc = ZEROS

        self.env = grid_env.GridEnv(grid_spec.spec_from_string(maze_str))

    def current_waypoint(self):
        return self._waypoints[self._waypoint_idx]

    def get_action(self, location, velocity, target):
        if np.linalg.norm(self._target - target) > 1e-3:
                self._new_target(location, target)
        # if self.openspace:
        #     if np.linalg.norm(self._target - target) > 1e-3:
        #         self._new_target(location, target)
        # elif np.linalg.norm(self._target - np.array(self.gridify_state(target))) > 0.8: 
        #     #print('New target!', target, 'old:', self._target)
        #     self._new_target(location, target)

        dist = np.linalg.norm(location - self._target)
        vel = self._waypoint_prev_loc - location
        vel_norm = np.linalg.norm(vel)
        task_not_solved = (dist >= self.solve_thresh) or (vel_norm >= self.vel_thresh)

        if task_not_solved:
            next_wpnt = self._waypoints[self._waypoint_idx]
        else:
            next_wpnt = self._target

        # Compute control
        prop = next_wpnt - location
        action = self.p_gain * prop + self.d_gain * velocity

        dist_next_wpnt = np.linalg.norm(location - next_wpnt)
        if task_not_solved and (dist_next_wpnt < self.solve_thresh) and (vel_norm<self.vel_thresh):
            self._waypoint_idx += 1
            if self._waypoint_idx == len(self._waypoints)-1:
                assert np.linalg.norm(self._waypoints[self._waypoint_idx] - self._target) <= self.solve_thresh, \
                         f"Distance was still {np.linalg.norm(self._waypoints[self._waypoint_idx] - self._target)} when last waypoint reached" + \
                         f"\nWaypoint: {self._waypoints[self._waypoint_idx]} | Target: {self._target}"

        self._waypoint_prev_loc = location
        # action = np.clip(action, -1.0, 1.0)
        return action, (not task_not_solved)

    def gridify_state(self, state):
        return (int(round(state[0])), int(round(state[1])))
    
    def _next_waypoint(self, w, delta, center):
        delta = np.array(delta, dtype = float)
        w = np.array(w, dtype = float)
        next_wpnt = center + delta
        return next_wpnt

    def _new_target(self, start, target):
        #print('Computing waypoints from %s to %s' % (start, target))
        #print("NEW CONTROLLER TARGET")
        if not self.openspace:
            start_grid = self.gridify_state(start)
            start_idx = self.env.gs.xy_to_idx(start_grid)
            target_grid = self.gridify_state(target)
            target_idx = self.env.gs.xy_to_idx(target_grid)
            self._waypoint_idx = 0
            self.env.gs[target_grid] = grid_spec.REWARD
            q_values = q_iteration.q_iteration(env=self.env, num_itrs=25, discount=0.99)
            # compute waypoints by performing a rollout in the grid
            max_ts = 100
            s = start_idx
            prev = start
            prev_grid = start_grid
            waypoints = []
            offset = np.random.uniform(low = -self.waypoint_noise, high = self.waypoint_noise, size = (2,))

            for i in range(max_ts):
                a = np.argmax(q_values[s])
                new_s, reward = self.env.step_stateless(s, a)
                waypoint_grid = self.env.gs.idx_to_xy(new_s)
                waypoint = self._next_waypoint(prev, waypoint_grid - prev_grid, prev_grid) + offset
                s = new_s
                if new_s == target_idx:
                    waypoints.append(target)
                    break
                else:
                    #waypoints.append((waypoint + prev)/2)
                    waypoints.append(waypoint)
                    prev = waypoint
                    prev_grid = waypoint_grid
            self.env.gs[target_grid] = grid_spec.EMPTY
            self._waypoints = waypoints
            self._waypoint_prev_loc = start
            self._target = target
        else:
            start = np.array(start)
            target = np.array(target)

            max_ts = 1
            t = np.linspace(0, 1, max_ts + 1)[1:].reshape((-1, 1))
            t[-1][0] = 1.0
            t = np.sort(t)

            noise = np.random.uniform(low = -1, high = 1, size=(max_ts, 2))*0.01
            noise[-1][0] = 0.0

            waypoints = start[None, :] * (1 - t) + target[None, :] * t #- noise
            waypoints = [p for p in waypoints]

            self._waypoints = waypoints
            self._waypoint_prev_loc = start
            self._waypoint_idx = 0
            self._target = target



if __name__ == "__main__":
    print(q_iteration.__file__)
    TEST_MAZE = \
            "######\\"+\
            "#OOOO#\\"+\
            "#O##O#\\"+\
            "#OOOO#\\"+\
            "######"
    controller = WaypointController(TEST_MAZE)
    start = np.array((1,1), dtype=np.float32)
    target = np.array((4,3), dtype=np.float32)
    act, done = controller.get_action(start, target)
    print('wpt:', controller._waypoints)
    print(act, done)
    import pdb; pdb.set_trace()
    pass

