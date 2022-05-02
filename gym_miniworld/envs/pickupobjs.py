import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Key

class PickupObjs(MiniWorldEnv):
    """
    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.
    """

    def __init__(self, size=12, num_objs=5, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs

        super().__init__(
            max_episode_steps=2500,
            **kwargs
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup+1)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )

        obj_types = [Ball, Box, Key]

        for obj in range(self.num_objs):
            obj_type = self.rand.choice(obj_types)
            color = self.rand.color()

            if obj_type == Box:
                self.place_entity(Box(color=color, size=0.9))
            if obj_type == Ball:
                self.place_entity(Ball(color=color, size=0.9))
            if obj_type == Key:
                self.place_entity(Key(color=color))

        self.place_agent()

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.agent.carrying:
            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None
            self.num_picked_up += 1
            reward = 1

            if self.num_picked_up == self.num_objs:
                done = True

        return obs, reward, done, info

class PickupObjsCustom(MiniWorldEnv):
    """
    Room with multiple objects. Objects disappear when picked up.

    This Environment allows for the relative
    rewards for each of the objects to be set via the "set_rewards" method

    This environment is also larger and has more total objects more inline
    with Borsa et al. 2019. The action space is changed to the movement
    actions,and objects are pickered up when the agent is in their proximity
    """

    def __init__(self, size=40, num_objs=30, object_rewards = {"ball" : 1, "box": 1, "key": -1}, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs
        self.object_rewards = object_rewards

        super().__init__(
            max_episode_steps=100,
            **kwargs
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )

        obj_types = [Ball, Box, Key, ]
        self.boxes = []
        self.balls = []
        self.keys = []


        # note that the number of total objects needs to be a multiple of 3
        n_each_obj = self.num_objs // 3
        _objs = [Ball] * n_each_obj + [Box] * n_each_obj + [Key] * n_each_obj
        for obj_type  in _objs:
            #obj_type = self.rand.choice(obj_types)
            #color = self.rand.color()
            if obj_type == Box:
                _entity = Box(color="blue", size=0.9)
                self.boxes.append(_entity)
            if obj_type == Ball:
                _entity = Ball(color="red", size=0.9)
                self.balls.append(_entity)
            if obj_type == Key:
                _entity = Key(color="yellow")
                self.keys.append(_entity)
            self.place_entity(_entity)

        self.place_agent()

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        feature = np.zeros([3])
        for box in self.boxes:
            if self.near(box):
                reward += self.object_rewards["box"]
                self.entities.remove(box)
                self.boxes.remove(box)
                feature[0] = 1

                self.num_picked_up += 1

        for ball in self.balls:
            if self.near(ball):
                reward += self.object_rewards["ball"]
                self.entities.remove(ball)
                self.balls.remove(ball)
                feature[1] = 1

                self.num_picked_up += 1

        for key in self.keys:
            if self.near(key):
                reward += self.object_rewards["key"]
                self.entities.remove(key)
                self.keys.remove(key)
                feature[2] = 1

                self.num_picked_up += 1

            if self.num_picked_up == self.num_objs:
                done = True

        info["feature"] = feature

        return obs, reward, done, info

    def switch(self, reward_dict = None):
        if reward_dict == None:
            # choose randomly
            pass
        else:
            self.object_rewards = reward_dict

    def reset(self):
        #self.switch({"ball" : 10, "box": 10, "key": 10})
        ob = super().reset()
        return ob
