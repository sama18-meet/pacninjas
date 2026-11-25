# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import heapq
import random
from collections import deque
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


SIGHT_RANGE = 5     # enemies within this Manhattan distance of us are visible
SONAR_NOISE = 6     # noisy distance readings are off by at most this much

MOVES = {
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST: (1, 0),
    Directions.WEST: (-1, 0),
}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """Food collector. Every turn it:
      1. updates its picture of the ghosts (visible, or remembered for a while),
      2. decides between collecting and going home (sticky until deposit),
      3. picks one target (food, or capsule), committed to,
      4. takes the first step of the cheapest route that avoids ghosts.

    Safety is the escape margin: for a cell, how many moves we are ahead of
    the fastest ghost to our best exit. Exits are the entry cells into our
    side, plus capsules. Food is acceptable only if the margin stays at least
    MIN_MARGIN once we stand on it, so a ghost near food is fine, but a ghost
    that stands between us and home is not."""

    INF = 9999
    MIN_MARGIN = 1        # food/capsule is acceptable if we reach an exit this many moves ahead
    TURN_HOME_MARGIN = 0  # carrying and margin at or below this: a ghost is cutting us off
    LOAD_MIN = 10          # carry this much (or a fifth of the starting food, if more): go home
    TIME_MARGIN = 6       # moves kept spare when racing home before the game ends
    SWITCH_SLACK = 1      # keep the current target unless another is this many moves closer
    NEAR_STEPS = 2        # route cells this close to an active ghost cost extra
    NEAR_COST = 10
    FORGET_TURNS = 4      # a ghost unseen for this many turns is ignored
    BLOCK_TURNS = 15      # a pair of cells we bounced between is avoided this long
    BLOCK_COST = 30
    RISK_TURNS = 6        # after a detected loop, ignore the no-ghost-adjacent rule this long
    VISIT_COST = 0.5      # extra route cost per recent visit (breaks loops)
    HUNT_STEPS = 2        # chase an invader this close, unless it stands on our border
    MEMORY = 20           # recent positions remembered

    # ------------------------------------------------------------ setup

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        walls = game_state.get_walls()
        self.walls = walls
        mid = walls.width // 2
        self.home_x = mid - 1 if self.red else mid
        self.home_border = [(self.home_x, y) for y in range(walls.height)
                            if not walls[self.home_x][y]]
        self.home_set = set(self.home_border)
        entries = [self.enemy_side_of(b) for b in self.home_border]
        self.entries = [e for e in entries if not walls[e[0]][e[1]]]
        # Our walking distance from every enemy-side cell to each entry (static)
        self.entry_maps = {e: self.enemy_bfs([e]) for e in self.entries}
        self.start_food = len(self.get_food(game_state).as_list())
        self.load_limit = max(self.LOAD_MIN, self.start_food // 5)
        self.mode = 'collect'
        self.target = None
        self.history = []
        self.visits = {}
        self.blocked = {}
        self.turn = 0
        self.risky_until = 0
        self.invaders = []
        self.forbid = set()
        self.near = set()
        self.belief = {}
        self.prev_enemy = {}
        self.unseen = {}
        self.init_tracking(game_state)

    def cell(self, pos):
        return (int(pos[0]), int(pos[1]))

    def is_home(self, c):
        return c[0] <= self.home_x if self.red else c[0] >= self.home_x

    def enemy_side_of(self, b):
        return (b[0] + 1, b[1]) if self.red else (b[0] - 1, b[1])

    def neighbours(self, c):
        x, y = c
        return [(x + dx, y + dy) for dx, dy in MOVES.values()
                if not self.walls[x + dx][y + dy]]

    def enemy_bfs(self, starts):
        """Steps from the nearest start cell, walking only on the enemy side
        (ghosts cannot step onto our side). Start cells are always included."""
        dist = {c: 0 for c in starts}
        queue = deque(starts)
        while queue:
            c = queue.popleft()
            for n in self.neighbours(c):
                if n not in dist and not self.is_home(n):
                    dist[n] = dist[c] + 1
                    queue.append(n)
        return dist

    # ------------------------------------------------------ enemy tracking
    #
    # Enemy positions are hidden beyond 5 squares. We keep, for every enemy,
    # the set of cells it could be in (its belief), updated every turn.

    def init_tracking(self, game_state):
        self.cells = [(x, y) for x in range(self.walls.width)
                      for y in range(self.walls.height) if not self.walls[x][y]]
        for i in self.get_opponents(game_state):
            self.belief[i] = {game_state.get_initial_agent_position(i)}
            self.prev_enemy[i] = (False, 0)
            self.unseen[i] = 0

    def filter_belief(self, cells, is_pacman, noisy, my_pos, team):
        """Keep cells consistent with what we know about a hidden enemy:
        on the right side, matching the noisy distance, out of sight."""
        out = set()
        for c in cells:
            if self.is_home(c) != is_pacman:
                continue
            if abs(util.manhattan_distance(c, my_pos) - noisy) > SONAR_NOISE:
                continue
            if any(util.manhattan_distance(c, t) <= SIGHT_RANGE for t in team):
                continue
            out.add(c)
        return out

    def track_enemies(self, game_state, pos):
        team = [game_state.get_agent_position(i) for i in self.get_team(game_state)]
        noisy = game_state.get_agent_distances()
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            e_pos = st.get_position()
            start = game_state.get_initial_agent_position(i)
            if e_pos is not None:
                self.belief[i] = {self.cell(e_pos)}
                self.unseen[i] = 0
            else:
                self.unseen[i] += 1
                new = set()
                for c in self.belief[i]:            # moved at most one step
                    new.add(c)
                    new.update(self.neighbours(c))
                was_pacman, was_scared = self.prev_enemy[i]
                if (was_pacman and not st.is_pacman) or (was_scared > 1 and st.scared_timer == 0):
                    new.add(start)                  # may have been eaten: respawned
                new = self.filter_belief(new, st.is_pacman, noisy[i], pos, team)
                if not new:
                    new = self.filter_belief(self.cells, st.is_pacman, noisy[i], pos, team)
                if not new:
                    new = {start}
                self.belief[i] = new
            self.prev_enemy[i] = (st.is_pacman, st.scared_timer)

    def threats_now(self, game_state):
        """Ghosts that can cut us off, as dicts with their reach map (steps to
        every enemy-side cell). Also collects visible invaders on our side."""
        out = []
        self.invaders = []
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            visible = st.get_position() is not None
            if st.is_pacman:
                if not visible:
                    continue
                c = self.cell(st.get_position())
                if c in self.home_set:
                    # An invader on our border can step back and become a ghost
                    out.append({'cells': {c}, 'visible': True, 'scared': 0,
                                'pacman': True, 'reach': self.enemy_bfs([c])})
                else:
                    self.invaders.append(c)
                continue
            if self.unseen[i] > self.FORGET_TURNS:
                continue
            cells = self.belief[i]
            out.append({'cells': cells, 'visible': visible, 'scared': st.scared_timer,
                        'pacman': False, 'reach': self.enemy_bfs(list(cells))})
        return out

    # ------------------------------------------------------- safety model

    def arrive(self, c, threats):
        """Earliest move at which a ghost can be dangerous on cell c: it must
        walk there AND its scared timer must have run out."""
        return min([max(t['reach'].get(c, self.INF), t['scared']) for t in threats]
                   or [self.INF])

    def exit_list(self, capsules, threats):
        """(our distance map, ghost arrival time) for every exit."""
        exits = [(m, self.arrive(e, threats)) for e, m in self.entry_maps.items()]
        exits += [(self.enemy_bfs([c]), self.arrive(c, threats)) for c in capsules]
        return exits

    def margin(self, c, exits):
        """How many moves we are ahead of the ghosts at the best exit. Positive
        means we can still get out first. Our side is always safe."""
        if self.is_home(c):
            return self.INF
        best = -self.INF
        for dist_map, ghost_time in exits:
            d = dist_map.get(c)
            if d is not None and ghost_time - d > best:
                best = ghost_time - d
        return best

    def hazards(self, threats):
        """forbid: cells we may never step on (next to a visible, active ghost).
        near: enemy-side cells close to an active ghost (routes avoid them)."""
        forbid, near = set(), set()
        for t in threats:
            if t['scared'] > 0:
                continue                             # scared ghosts can't eat us
            near.update(c for c, d in t['reach'].items() if d <= self.NEAR_STEPS)
            if t['visible']:
                for g in t['cells']:
                    forbid.add(g)
                    forbid.update(self.neighbours(g))
        return forbid, near

    # ---------------------------------------------------------- routing

    def step_cost(self, c):
        cost = 1 + self.VISIT_COST * self.visits.get(c, 0)
        if c in self.near:
            cost += self.NEAR_COST
        if self.blocked.get(c, 0) > self.turn:
            cost += self.BLOCK_COST
        return cost

    def route_tree(self, game_state, pos, strict=True):
        """Cheapest routes from pos to every reachable cell (Dijkstra).
        strict=False ignores the never-next-to-a-ghost rule (last resort).
        Returns cell -> (cost, first action, parent, steps)."""
        forbid = self.forbid if strict else set()
        tree = {}
        heap = []
        counter = 0
        actions = [a for a in game_state.get_legal_actions(self.index)
                   if a != Directions.STOP]
        random.shuffle(actions)                      # random tie-breaking
        for a in actions:
            dx, dy = MOVES[a]
            c = (pos[0] + dx, pos[1] + dy)
            if c in forbid:
                continue
            counter += 1
            heapq.heappush(heap, (self.step_cost(c), counter, 1, c, a, pos))
        done = {pos}
        while heap:
            cost, _, steps, c, first, parent = heapq.heappop(heap)
            if c in done:
                continue
            done.add(c)
            tree[c] = (cost, first, parent, steps)
            for n in self.neighbours(c):
                if n in done or n in forbid:
                    continue
                counter += 1
                heapq.heappush(heap, (cost + self.step_cost(n), counter,
                                      steps + 1, n, first, c))
        return tree

    def first_step(self, tree, goals):
        cands = [g for g in goals if g in tree]
        if not cands:
            return None
        return tree[min(cands, key=lambda g: tree[g][0])][1]

    def pick_target(self, cands, tree):
        """Keep the current target unless another is clearly closer."""
        best = min(cands, key=lambda c: tree[c][0])
        if self.target in cands and tree[self.target][0] <= tree[best][0] + self.SWITCH_SLACK:
            return self.target
        return best

    def eat_scared(self, me, threats, tree):
        """A scared ghost right next to us is free points: step into it."""
        if not me.is_pacman:
            return None
        for t in threats:
            if t['visible'] and not t['pacman'] and t['scared'] > 0:
                g = next(iter(t['cells']))
                if g in tree and tree[g][3] == 1:
                    return tree[g][1]
        return None

    def hunt(self, me, tree):
        """On our side and not scared: chase a close invader, unless it stands
        on our border (it would just step back over)."""
        if me.is_pacman or me.scared_timer > 0:
            return None
        close = [c for c in self.invaders if c in tree and tree[c][3] <= self.HUNT_STEPS]
        if not close:
            return None
        return tree[min(close, key=lambda c: tree[c][3])][1]

    def flee(self, game_state, pos, threats):
        """Last resort: the legal move that keeps the most distance from ghosts."""
        best, best_score = Directions.STOP, -self.INF
        for a in game_state.get_legal_actions(self.index):
            if a == Directions.STOP:
                continue
            dx, dy = MOVES[a]
            c = (pos[0] + dx, pos[1] + dy)
            score = min([t['reach'].get(c, self.INF) for t in threats
                         if t['scared'] == 0] or [self.INF])
            if c in self.forbid:
                score -= 1000
            if score > best_score:
                best, best_score = a, score
        return best

    def update_loop(self, pos):
        """Remember recent cells (route cost) and break two-cell bounces."""
        self.history.append(pos)
        self.history = self.history[-self.MEMORY:]
        self.visits = {}
        for c in self.history:
            self.visits[c] = self.visits.get(c, 0) + 1
        last = self.history[-4:]
        if (len(last) == 4 and last[0] == last[2] and last[1] == last[3]
                and last[0] != last[1]):
            for c in (last[0], last[1]):
                self.blocked[c] = self.turn + self.BLOCK_TURNS
            # A ghost shadowing us makes the safe routes flip every turn. Push
            # through for a few turns instead (a lost pellet beats a standoff).
            self.risky_until = self.turn + self.RISK_TURNS
            self.target = None

    # --------------------------------------------------------- decision

    def choose_action(self, game_state):
        self.turn += 1
        me = game_state.get_agent_state(self.index)
        pos = self.cell(me.get_position())
        carrying = me.num_carrying
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)

        self.track_enemies(game_state, pos)
        self.update_loop(pos)
        threats = self.threats_now(game_state)
        self.forbid, self.near = self.hazards(threats)
        exits = self.exit_list(capsules, threats)
        tree = self.route_tree(game_state, pos, strict=self.turn > self.risky_until)
        goals = self.home_border + capsules + food
        if not any(g in tree for g in goals):
            # Boxed in by ghosts: accept some risk rather than oscillate
            tree = self.route_tree(game_state, pos, strict=False)

        home_cost = min([tree[h][0] for h in self.home_border if h in tree] or [self.INF])
        reachable_food = [f for f in food if f in tree]
        food_ok = [f for f in reachable_food if self.margin(f, exits) >= self.MIN_MARGIN]
        food_cost = min([tree[f][0] for f in food_ok] or [self.INF])
        moves_left = game_state.data.timeleft // 4

        # Enemy state is readable even when its position is hidden: how much
        # food it holds (num_carrying) is food it will still bank. If our lead
        # is bigger than everything the enemy can still score, the result is
        # decided, so food we carry is not worth risking.
        enemy_carry = sum(game_state.get_agent_state(i).num_carrying
                          for i in self.get_opponents(game_state))
        enemy_max_gain = len(self.get_food_you_are_defending(game_state).as_list()) + enemy_carry
        decided = self.get_score(game_state) > enemy_max_gain

        # ---- Mode: collect, or return (sticky until the food is deposited)
        if carrying == 0:
            self.mode = 'collect'
        elif decided:
            self.mode = 'return'
        elif self.mode != 'return':
            if (self.margin(pos, exits) < self.TURN_HOME_MARGIN   # a ghost is cutting us off
                    or carrying >= self.load_limit               # big load: bank it
                    or not food_ok                               # nothing safe left
                    or food_cost > home_cost                     # next food farther than home
                    or home_cost + self.TIME_MARGIN >= moves_left  # time running out
                    or len(food) <= 2                            # game almost won
                    or any(0 < t['scared'] <= home_cost + 3 for t in threats)):
                self.mode = 'return'
        if not food:
            self.mode = 'return'

        # ---- Action
        action = None
        if self.mode == 'return':
            action = self.first_step(tree, self.home_border) or self.first_step(tree, capsules)
        else:
            action = self.eat_scared(me, threats, tree) or self.hunt(me, tree)
            if action is None:
                caps = [c for c in capsules
                        if c in tree and self.margin(c, exits) >= self.MIN_MARGIN]
                cands = food_ok + caps
                if cands:
                    self.target = self.pick_target(cands, tree)
                    action = tree[self.target][1]
                elif reachable_food:
                    # No safe food anywhere: take the least risky food
                    best = max(reachable_food,
                               key=lambda f: (self.margin(f, exits), -tree[f][0]))
                    action = tree[best][1]
                else:
                    action = self.first_step(tree, self.home_border)
        if action is None:
            action = self.flee(game_state, pos, threats)
        return action


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}