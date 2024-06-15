import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData 
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BACK_WALL_Y, BLUE_GOAL_BACK, ORANGE_GOAL_BACK, BALL_RADIUS
from rlgym_ppo.util import MetricsLogger
from lookup_act import LookupAction
from abc import abstractmethod
from rlgym_sim.utils import math
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, BALL_MAX_SPEED

SUPERSONIC_THRESHOLD = 2200

max_vel_diff = 3000
# This is a zeo sum reward function that rewards the player that scored the goal
#class ZeroSumGoalReward(RewardFunction):
#    def init(self):
#        super().init()
#        self.prev_score_blue = 0
#        self.prev_score_orange = 0
#        self.prev_state = None
#        # weights add up to .5
#        self.height = 0.35
#        self.speed = 0.15
#        # max values for rewards
#        self.max_speed = 135 * KPH_TO_VEL
#        self.max_dis = 2500
#
#        self.reward = np.zeros(2)
#
#    def reset(self, initial_state: GameState):
#        self.prev_score_orange = initial_state.orange_score
#        self.prev_state = initial_state
#
#    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
#        score = state.blue_score
#
#        blue_rew = 0
#        orange_rew = 0
#
#        scored = False
#
#        if self.prev_score_blue < score:
#            blue_rew = (np.linalg.norm(self.prev_state.ball.linear_velocity) / self.max_speed) * self.speed + 0.5
#            blue_rew += self.prev_state.ball.position[2] / (642 - BALL_RADIUS) * self.height
#
#        score_orange = state.orange_score
#        # check to see if goal scored
#        if score_orange > self.prev_score_orange:
#            orange_rew = (np.linalg.norm(self.prev_state.ball.linear_velocity) / self.max_speed) * self.speed + 0.5
#            orange_rew += self.prev_state.ball.position[2] / (642 - BALL_RADIUS) * self.height
#
#        self.reward[0] = blue_rew - orange_rew
#        self.reward[1] = orange_rew - blue_rew
#
#        self.prev_score_blue = score
#        self.prev_score_orange = score_orange
#        self.prev_state = state
#
#        blue_reward = self.reward[0]
#        orange_reward = self.reward[1]
#
#        if player.team_num == BLUE_TEAM:
#            return blue_reward
#
#        elif player.team_num == BLUE_TEAM:
#            return orange_reward
#
#        return 0

class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0

class InAirReward(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if not player.on_ground:
            if player.ball_touched:
                return 2
            else:
                return 1
        else:
            return 0
        
class GoalSpeedReward(RewardFunction):
    def __init__(self):
        self.prev_score_blue = 0
        self.prev_score_orange = 0
        self.prev_state_blue = None
        self.prev_state_orange = None
    def reset(self, initial_state: GameState):
        self.prev_score_blue = initial_state.blue_score
        self.prev_score_orange = initial_state.orange_score
        self.prev_state_blue = initial_state
        self.prev_state_orange = initial_state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            score = state.blue_score
            # check to see if goal scored
            if score > self.prev_score_blue:
                reward = np.linalg.norm(self.prev_state_blue.ball.linear_velocity) / BALL_MAX_SPEED
                self.prev_state_blue = state
                self.prev_score_blue = score
                return reward
            else:
                self.prev_state_blue = state
                return 0.0
        else:
            score = state.orange_score
            # check to see if goal scored
            if score > self.prev_score_orange:
                reward = np.linalg.norm(self.prev_state_orange.ball.linear_velocity) / BALL_MAX_SPEED
                self.prev_state_orange = state
                self.prev_score_orange = score
                return reward
            else:
                self.prev_state_orange = state
                return 0.0            

class TouchVelChange(RewardFunction):
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = initial_state.ball.linear_velocity

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        
        if player.ball_touched and (state.ball.linear_velocity > self.last_vel).all():
            vel_difference = np.linalg.norm(self.last_vel - state.ball.linear_velocity)

            if vel_difference >= 150:
                reward = np.sqrt(min(vel_difference,max_vel_diff)/max_vel_diff) 

        self.last_vel = state.ball.linear_velocity

        return max(0, reward)
    

class GoalSpeedAndPlacementReward(RewardFunction):
    def __init__(self):
        self.prev_score_blue = 0
        self.prev_score_orange = 0
        self.prev_state_blue = None
        self.prev_state_orange = None
        self.min_height = BALL_RADIUS + 10
        self.height_reward = 1.75

    def reset(self, initial_state: GameState):
        self.prev_score_blue = initial_state.blue_score
        self.prev_score_orange = initial_state.orange_score
        self.prev_state_blue = initial_state
        self.prev_state_orange = initial_state

    
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            score = state.blue_score
            # check to see if goal scored
            if score > self.prev_score_blue:
                reward = np.linalg.norm(self.prev_state_blue.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_blue.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_blue = state
                self.prev_score_blue = score
                return reward
            else:
                self.prev_state_blue = state
                return 0.0
        else:
            score = state.orange_score
            # check to see if goal scored
            if score > self.prev_score_orange:
                reward = np.linalg.norm(self.prev_state_orange.ball.linear_velocity) / BALL_MAX_SPEED
                if self.prev_state_orange.ball.position[2] > self.min_height:
                    reward = self.height_reward * reward
                self.prev_state_orange = state
                self.prev_score_orange = score
                return reward
            else:
                self.prev_state_orange = state
                return 0.0

class RewardIfTouchedLast(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> bool:
        return state.last_touch == player.car_id
    
class BallYCoordinateReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == BLUE_TEAM:
            return (state.ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent
        else:
            return (state.inverted_ball.position[1] / (BACK_WALL_Y + BALL_RADIUS)) ** self.exponent

class BehindBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.team_num == BLUE_TEAM and player.car_data.position[1] < state.ball.position[1] \
            or player.team_num == ORANGE_TEAM and player.car_data.position[1] > state.ball.position[1]:
            return 1
        else:
            return -1

class InAirReward(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if not player.on_ground:
            return 0.5
        else:
            return -0.1
        

class AlignBallGoal(RewardFunction):
    def __init__(self, defense=1.0, offense=1.0):
        super().__init__()
        self.defense = defense
        self.offense = offense

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        ball = state.ball.position
        pos = player.car_data.position
        protecc = np.array(BLUE_GOAL_BACK)
        attacc = np.array(ORANGE_GOAL_BACK)
        if player.team_num == ORANGE_TEAM:
            protecc, attacc = attacc, protecc

        # Align player->ball and net->player vectors
        defensive_reward = self.defense * math.cosine_similarity(ball - pos, pos - protecc)

        # Align player->ball and player->net vectors
        offensive_reward = self.offense * math.cosine_similarity(ball - pos, attacc - pos)

        return defensive_reward + offensive_reward

class MoreFlipsReward(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.boost_amount < 25:
            if not player.has_flip:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

class MaintainAboveZeroBoostReward(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.boost_amount < 1:
            return -1
        else:
            if player.boost_amount < 5:
                return -0.5
            else:
                if player.boost_amount < 15:
                    return -0.1
                else:
                   return 0

class SpeedTowardBallReward(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_vel = player.car_data.linear_velocity
        pos_diff = (state.ball.position - player.car_data.position)
        dist_to_ball = np.linalg.norm(pos_diff)
        dir_to_ball = pos_diff / dist_to_ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            return 0
    
class MaintainHighSpeedReward(RewardFunction):
    def __init__(self):
        super().__init__()
    def reset(self, initial_state: GameState):
        pass
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        player_vel = np.linalg.norm(player.car_data.linear_velocity)

        if player_vel > SUPERSONIC_THRESHOLD:
            return 1
        else:
            if player_vel > SUPERSONIC_THRESHOLD * 0.8:
                return 0.7
            else:
                if player_vel > SUPERSONIC_THRESHOLD * 0.6:
                    return 0.5
                else:
                    if player_vel > SUPERSONIC_THRESHOLD * 0.4:
                        return 0.1
                    else:
                        if player_vel > SUPERSONIC_THRESHOLD * 0.2:
                            return -0.2
                        else:
                            return -0.5

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, FaceBallReward, VelocityBallToGoalReward, \
        EventReward
    from rlgym_sim.utils.obs_builders import AdvancedObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 15
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    rewards_to_combine = (MoreFlipsReward(),
                          JumpTouchReward(),
                          InAirReward(),
                          SpeedTowardBallReward(),
                          FaceBallReward(),
                          BehindBallReward(),
                          RewardIfTouchedLast(),
                          VelocityPlayerToBallReward(),
                          BallYCoordinateReward(),
                          AlignBallGoal(),
                          VelocityBallToGoalReward(),
                          GoalSpeedReward(),
                          GoalSpeedAndPlacementReward(),\
                          EventReward(team_goal=4000.00, shot=200.00, concede=-300.00, demo=200.00, boost_pickup=30.00))
    reward_weights = (1.50, 20.00, 1.00, 2.00, 8.00, 1.50, 2.50, 2.00, 0.40, 12.00, 10.00, 12.00, 15.00, 20.00)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    obs_builder = AdvancedObs()

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser)
    import rocketsimvis_rlgym_sim_client as rsv
    type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = ExampleLogger()
    n_proc = 128
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    import os

    learner = Learner(build_rocketsim_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=metrics_logger,
        ppo_batch_size=200000, #this to ts_per_iteration
        ts_per_iteration=200000, #50k = early, 100k = hitting the ball, shooting and scoring 200k or 300k
        policy_layer_sizes=(2048, 1024, 512, 512), #If you change this you have to restart the training
        critic_layer_sizes=(2048, 1024, 512, 512), #Just Match the poloicy layer sizes
        exp_buffer_size=600000, # this to ts_per_iteration*2 or ts_per_iteration*3
        ppo_minibatch_size=50000, #small portion of ppo_batch_size, I recommend 25_000 or 50_000
        ppo_ent_coef=0.01, #near 0.01.
        ppo_epochs=2, #I recommend 2 or 3, this is how much it goes over the same data
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=400_000,
        timestep_limit=100_000_000_000,
        log_to_wandb=True)
    learner.learn()