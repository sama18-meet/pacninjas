import random

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from types import SimpleNamespace

def create_team(first_index, second_index, is_red, **args):
    return [SmartAgent(first_index), SmartAgent(second_index)]

class TestEvaluatorAgent(CaptureAgent):


    # calculated per each egent 
    def evaluate_risk_and_opportunity(agent, game_state):
        my_state = game_state.get_agent_state(agent.index)
        my_pos = nearest_point(my_state.get_position())
        opponents = agent.get_opponents(game_state)
        enemies = [game_state.get_agent_state(i) for i in opponents]
        noisy_dists = game_state.get_agent_distances()

        # Update known enemy positions (automatically clears enemies who left or were eaten)
        known_enemy_pos = agent.update_and_get_enemy_positions(game_state)
        prev_state = agent.get_previous_observation()

        # --- 1. CALCULATE RISK ---
        risk = my_state.num_carrying * 10

        for i, enemy in zip(opponents, enemies):
            # A. Check if enemy is invading our side
            if enemy.is_pacman:
                risk += 30
                if i in known_enemy_pos:
                    invader_pos = nearest_point(known_enemy_pos[i])
                    invader_dist = agent.get_maze_distance(my_pos, invader_pos)
                    risk += max(0, 25 - invader_dist)

            # B. Check if an invader JUST got eaten or left our territory since last turn
            elif prev_state is not None and prev_state.get_agent_state(i).is_pacman:
                # Enemy was on our side last turn, now is_pacman is False -> threat cleared!
                risk = max(0, risk - 20)

            # C. Check active enemy defenders (ghosts) near us when we are on offense
            if not enemy.is_pacman and enemy.scared_timer == 0:
                if i in known_enemy_pos:
                    ghost_pos = nearest_point(known_enemy_pos[i])
                    dist = agent.get_maze_distance(my_pos, ghost_pos)
                else:
                    dist = noisy_dists[i]

                if dist <= 5:
                    risk += (6 - dist) * 20

        # --- 2. CALCULATE OPPORTUNITY ---
        opportunity = 0
        food_list = agent.get_food(game_state).as_list()
        if food_list:
            min_food = min(agent.get_maze_distance(my_pos, f) for f in food_list)
            opportunity += max(0, 30 - min_food * 2)

        if any(e.scared_timer > 5 for e in enemies):
            opportunity += 50

        score = agent.get_score(game_state)
        if score < 0:
            opportunity += abs(score) * 5

        return risk, opportunity

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        # Stores {enemy_index: (x, y)} across turns
        self.last_enemy_pos = {}
        self.start = game_state.get_agent_position(self.index) # get the initial state of the agent
        self.boundary = self.get_home_boundary(game_state) #get the safe zone

    ## CALCULATE BOUNDARY
    def get_home_boundary(self, game_state):
        layout = game_state.data.layout
        walls = game_state.get_walls()
        mid_x = layout.width // 2

        # Red team owns the left half (x = mid_x - 1); Blue owns the right half (x = mid_x)
        boundary_x = mid_x - 1 if self.red else mid_x

        boundary_cells = []
        for y in range(1, layout.height - 1):
            if not walls[boundary_x][y]:
                boundary_cells.append((boundary_x, y))
        return boundary_cells



    def update_and_get_enemy_positions(self, game_state):
        opponents = self.get_opponents(game_state)

        # 1. Always update with directly visible enemies
        for i in opponents:
            pos = game_state.get_agent_position(i)
            if pos is not None:
                self.last_enemy_pos[i] = pos
            # If enemy is no longer invading (retreated or got eaten), clear old memory
            elif not game_state.get_agent_state(i).is_pacman and i in self.last_enemy_pos:
                del self.last_enemy_pos[i]

        # 2. If food disappeared on our side, overwrite with that exact coordinate
        prev_state = self.get_previous_observation()
        if prev_state is not None:
            prev_food = set(self.get_food_you_are_defending(prev_state).as_list())
            curr_food = set(self.get_food_you_are_defending(game_state).as_list())
            eaten_food = list(prev_food - curr_food)

            if eaten_food:
                invisible_invaders = [
                    i for i in opponents
                    if game_state.get_agent_position(i) is None and game_state.get_agent_state(i).is_pacman
                ]
                for i, food_pos in zip(invisible_invaders, eaten_food):
                    self.last_enemy_pos[i] = food_pos

        # 3. Return all known/remembered positions (any enemy not in this dict falls back to noisy_dists)
        return self.last_enemy_pos




class SmartAgent(TestEvaluatorAgent):

    def choose_action(self, game_state):

        known_enemy_pos = self.update_and_get_enemy_positions(game_state)
        risk, opportunity = self.evaluate_risk_and_opportunity(game_state)

        # 2. PLANNER: Pick strategy & target (x, y) coordinate on the map
        strategy = self.decide_strategy(game_state, risk, opportunity)
        target_pos = self.get_objective_target(game_state, strategy, risk, known_enemy_pos)

        # Optional: Highlight the target square on the GUI so you can visually verify!
        self.debug_draw(target_pos, [0, 1, 0], clear=True)

        # 3. EXECUTION: Pick legal action that minimizes maze distance to target_pos
        actions = game_state.get_legal_actions(self.index)
        best_action = min(
            actions,
            key=lambda a: self.get_maze_distance(
                nearest_point(game_state.generate_successor(self.index, a).get_agent_position(self.index)),
                target_pos
            )
        )
        return best_action


    def get_closest_retreat_point(self, my_pos):
        """Returns the closest walkable boundary square to safely bank points."""
        return min(self.boundary, key=lambda b: self.get_maze_distance(my_pos, b))
    # =========================================================================
    # 2. DECISION TREE (Evaluates top-to-bottom priority rules)
    # =========================================================================
    def decide_strategy(self, game_state, risk, opportunity):
        my_state = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman]
        food_left = len(self.get_food(game_state).as_list())

        # Priority 1: Game almost won (<= 2 dots left) OR carrying enough dots + high risk -> RETREAT
        if food_left <= 2 or (my_state.num_carrying > 0 and risk >= 65) or my_state.num_carrying >= 5:
            return 'RETREAT'

        # Priority 2: Invaders on our side and risk outweighs opportunity -> DEFEND
        if len(invaders) > 0 and risk > opportunity:
            return 'DEFEND'

        # Priority 3: High risk while empty-handed -> fall back to DEFEND/PATROL
        if risk >= 75:
            return 'DEFEND'

        # Priority 4: Default when safe or good opportunity -> ATTACK
        return 'ATTACK'

    # =========================================================================
    # 3. OBJECTIVE TARGET SELECTOR (Finds the exact (x, y) goal on the map)
    # =========================================================================
    def get_objective_target(self, game_state, strategy, risk, known_enemy_pos):
        my_pos = nearest_point(game_state.get_agent_position(self.index))

        # --- GOAL A: RETREAT ---
        if strategy == 'RETREAT':
            return self.get_closest_retreat_point(my_pos)

        # --- GOAL B: DEFEND (Predict enemy direction & intercept) ---
        elif strategy == 'DEFEND':
            return self.get_predicted_invader_target(game_state, my_pos, known_enemy_pos)

        # --- GOAL C: ATTACK (Food within a safe radius from the border) ---
        else:
            return self.get_safe_attack_target(game_state, my_pos, risk)

    def get_predicted_invader_target(self, game_state, my_pos, known_enemy_pos):
        """
        Finds the closest invader and predicts 1-2 steps ahead along their movement vector.
        """
        opponents = self.get_opponents(game_state)
        invader_ids = [i for i in opponents if game_state.get_agent_state(i).is_pacman and i in known_enemy_pos]

        if not invader_ids:
            # No visible/tracked invader -> patrol the center of our boundary
            return self.boundary[len(self.boundary) // 2]

        # Pick the closest known invader
        target_id = min(invader_ids, key=lambda i: self.get_maze_distance(my_pos, nearest_point(known_enemy_pos[i])))
        curr_enemy_pos = nearest_point(known_enemy_pos[target_id])

        # Predict direction using previous observation if available
        prev_state = self.get_previous_observation()
        if prev_state is not None:
            prev_enemy_pos = prev_state.get_agent_position(target_id)
            if prev_enemy_pos is not None:
                prev_enemy_pos = nearest_point(prev_enemy_pos)
                dx = curr_enemy_pos[0] - prev_enemy_pos[0]
                dy = curr_enemy_pos[1] - prev_enemy_pos[1]

                # Project 2 steps ahead along their velocity vector if walkable
                walls = game_state.get_walls()
                for step in (2, 1):
                    pred_x = curr_enemy_pos[0] + dx * step
                    pred_y = curr_enemy_pos[1] + dy * step
                    if 0 <= pred_x < walls.width and 0 <= pred_y < walls.height and not walls[pred_x][pred_y]:
                        return (pred_x, pred_y)

        return curr_enemy_pos

    def get_safe_attack_target(self, game_state, my_pos, risk):
        """
        Selects the closest enemy food within an 'attack radius' from our border.
        Higher risk shrinks the allowed radius so the agent stays near the exit.
        """
        food_list = self.get_food(game_state).as_list()
        if not food_list:
            return self.get_closest_retreat_point(my_pos)

        # Dynamic attack radius based on risk (e.g., risk 0 -> radius 12; risk 50 -> radius 5)
        max_depth = max(4, 12 - int(risk // 7))

        # Filter food that is within `max_depth` steps of our home boundary
        safe_food = [
            f for f in food_list
            if min(self.get_maze_distance(f, b) for b in self.boundary) <= max_depth
        ]

        # If safe food exists in our radius, target the closest one; otherwise target closest overall
        candidates = safe_food if safe_food else food_list
        return min(candidates, key=lambda f: self.get_maze_distance(my_pos, f))