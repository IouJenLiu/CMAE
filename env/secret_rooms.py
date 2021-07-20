import numpy as np
import copy
from env.rooms import Rooms, Entity, Door
import gym
TOP = 0
BOT = 1
LEFT = 2
RIGHT = 3


class SecretRooms(Rooms):
    def __init__(self, H=300, grid_size=25, n_actions=4, n_agents=2, checkpoint=False):
        super(SecretRooms, self).__init__(H, grid_size, n_actions, n_agents)
        # (x1, y1, x2, y2, door1_opened, door2_opened, door3_opened)
        self.observation_space = gym.spaces.MultiDiscrete([grid_size, grid_size, grid_size, grid_size, 8])
        onethrid, twothrid = grid_size // 3, 2 * grid_size // 3
        half = grid_size // 2
        one = grid_size
        door1_y, door2_y, door3_y = onethrid // 2, (onethrid + twothrid) // 2, (twothrid + one) // 2
        point8, point2 = int(grid_size * 0.8), int(grid_size * 0.2)

        self.init_doors = [Door(half, door1_y), Door(half, door2_y), Door(half, door3_y)]
        self.switches = [Entity(point2, point8), Entity(point8, door1_y),
                         Entity(point8, door2_y), Entity(point8, door3_y)]
        self.init_wall_map = np.zeros((grid_size, grid_size))
        self.init_wall_map[:, half] = 1
        self.init_wall_map[onethrid, half:] = 1
        self.init_wall_map[twothrid, half:] = 1
        self.doors = None
        self.checkpoint = checkpoint
        self.n_ckpt_bins = 5
        self.ckpt_bins = {'left_switch': np.linspace(2.5, self._dist(self.init_agents[0], self.switches[0]) - 1, self.n_ckpt_bins),
                          'door': np.linspace(4, self._dist(self.init_agents[0], self.init_doors[0]) - 1, self.n_ckpt_bins),
                          'right_switch': np.linspace(4, self._dist(self.init_doors[0], self.switches[1]) - 1, self.n_ckpt_bins)}
        self.ckpts = {'left_switch': [[False, bin] for bin in self.ckpt_bins['left_switch']],
                      'door': [[False, bin] for bin in self.ckpt_bins['door']],
                      'right_switch': [[False, bin] for bin in self.ckpt_bins['right_switch']]}
        self.success_rew = 3 if self.checkpoint else 1

    def reset(self):
        self.agents = copy.deepcopy(self.init_agents)
        self.wall_map = copy.deepcopy(self.init_wall_map)
        self.doors = copy.deepcopy(self.init_doors)
        self.step_count = 0
        self.done = False
        self.ckpts = {'left_switch': [[False, bin] for bin in self.ckpt_bins['left_switch']],
                      'door': [[False, bin] for bin in self.ckpt_bins['door']],
                      'right_switch': [[False, bin] for bin in self.ckpt_bins['right_switch']]}
        return np.array([self.agents[0].x, self.agents[0].y, self.agents[1].x, self.agents[1].y,
                         self.doors[0].open * 2**2 + self.doors[1].open * 2 + self.doors[2].open])

    def step(self, action):
        assert not self.done, "error: Trying to call step() after an episode is done"
        obs = []
        for agent_id, agent in enumerate(self.agents):
            self._update_agent_location(agent_id, action[agent_id])
            obs.extend([agent.x, agent.y])
        self._update_door_status()
        obs.append(self.doors[0].open * 2**2 + self.doors[1].open * 2 + self.doors[2].open)
        self.step_count += 1
        rew = self._reward()
        self.done = True if self.step_count == self.H or rew >= self.success_rew else False

        return np.array(obs), rew, self.done

    def _update_door_status(self):
        door_radius = 1
        self.doors = copy.deepcopy(self.init_doors)
        self.wall_map = copy.deepcopy(self.init_wall_map)
        for i, switch in enumerate(self.switches):
            for agent in self.agents:
                if self._dist(agent, switch) <= 1.5 * door_radius:
                    if i == 0:
                        for door in self.doors:
                            door.open = 1
                            self.wall_map[door.y - door_radius : door.y + door_radius + 1, door.x] = 0
                    else:
                        self.doors[i - 1].open = 1
                        self.wall_map[self.doors[i - 1].y - door_radius: self.doors[i - 1].y + door_radius + 1, self.doors[i - 1].x] = 0
                    return

    def _reward(self):
        rew = 0
        if self.checkpoint:
            rew += self._checkpoint_rew()
        for agent in self.agents:
            if agent.x < (self.grid_size / 2 + 1) or agent.y > (self.grid_size // 3 + 1):
                return rew
        rew += self.success_rew
        return rew

    def _checkpoint_rew(self):
        rew = 0

        if self.doors[0].open:
            for ckpt in self.ckpts['door']:
                agent = self.agents[0]
                #for agent in self.agents:
                if ckpt[0]:
                    continue
                if agent.x < (self.grid_size / 2 + 1):
                    if self._dist(agent, self.doors[0]) <= ckpt[1]:
                        rew += 0.1
                        ckpt[0] = True
        
        # Right switch
        for ckpt in self.ckpts['right_switch']:
            agent = self.agents[0]
            #for agent in self.agents:
            if ckpt[0]:
                continue
            if agent.x >= (self.grid_size / 2 + 1):
                if self._dist(agent, self.switches[1]) <= ckpt[1]:
                    rew += 0.1
                    ckpt[0] = True
        return rew