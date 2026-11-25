# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
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


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    Upgraded defender ("Vibes Defense").

    Strategy pillars (MECE):
      1. KNOW  - track every opponent with an exact belief grid (Bayes filter),
                 fed by the extra info the contest leaks even when an enemy is
                 hidden: is_pacman, num_carrying, num_returned, plus eaten food,
                 eaten capsules, sight range and noisy sonar distances.
      2. DECIDE - pick a mode:  HUNT (invaders on our side) or GUARD (none).
      3. ACT   - HUNT: cut off the invader on its own path (to its next food or
                 to its exit) instead of chasing its tail; block capsules;
                 shadow at distance 2 while we are scared.
                 GUARD: goalkeeper spot - the cell from which the attacker can
                 reach none of our food much before we can (minimax over food),
                 ignoring campers (ghosts that park or hover near the border
                 and never enter) and never standing in a dead-end pocket.
      4. SAFETY - never cross into enemy land, never touch an invader while
                 scared, fall back to the original reflex logic on any error.
                 All of OUR distances are measured inside our own half, so the
                 agent never freezes on a border cell.
      5. SCORE - when nobody is attacking, make short hit-and-run raids on
                 pellets just over the border that it can provably grab and
                 bring back before any enemy ghost can interfere (turns 0-0
                 ties into wins). Any invader cancels the raid at once.
    """

    SIGHT_RANGE = 5          # enemies within this Manhattan distance are visible
    SONAR_NOISE = 6          # noisy distance = true Manhattan distance +/- 6
    CARRY_ESCAPE = 3         # an invader carrying this much is assumed to head home
    SHADOW_TRIGGER = 6       # ghost this close to our border -> mirror it
    CAMP_LIMIT = 6           # ghost parked within 2 of our border this long = camper
    HOVER_LIMIT = 15         # ghost hovering within SHADOW_TRIGGER this long = camper
    DEFENDER_AFTER = 40      # never entered after this many turns while its mate did
    AMBUSH_DEPTH = 2         # guard at least this many steps inside our half
    GUARD_BAND = 5           # ...and at most this many (keeps the search cheap)
    RAID = True              # allow safe hit-and-run raids when our side is quiet
    RAID_TRIP = 18           # max steps: us -> enemy pellet -> back over the border
    RAID_CARRY = 3           # pellets to grab before heading home (more if scared)
    RAID_SAFETY = 3          # enemy ghost must be this many steps later than us
    RAID_RISK = 0.15         # max belief mass of a ghost that close to the pellet
    DEBUG = False

    # ------------------------------------------------------------------ #
    # Setup                                                               #
    # ------------------------------------------------------------------ #
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)

        walls = game_state.get_walls()
        self.width, self.height = walls.width, walls.height
        self.mid_x = self.width // 2
        self.legal_cells = walls.as_list(False)

        # Border columns: last column of our half and first column of theirs
        self.home_border_x = self.mid_x - 1 if self.red else self.mid_x
        self.enemy_border_x = self.mid_x if self.red else self.mid_x - 1
        self.home_cells = [p for p in self.legal_cells if self._is_home(p)]
        self.home_border = [p for p in self.legal_cells if p[0] == self.home_border_x]
        self.enemy_border = [p for p in self.legal_cells if p[0] == self.enemy_border_x]

        # Movement graph (STOP is allowed, so a cell is its own neighbour)
        self.neighbors = {}
        for (x, y) in self.legal_cells:
            nbrs = [(x, y)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and not walls[nx][ny]:
                    nbrs.append((nx, ny))
            self.neighbors[(x, y)] = nbrs

        # OUR travel distances, never leaving our half. The normal maze distance
        # may route through enemy land, which the defender refuses to enter;
        # using it made the agent freeze on border cells (every legal move looked
        # worse than STOP). Enemy travel still uses the normal maze distance.
        self.home_set = {p for p in self.legal_cells if self._is_home(p)}
        self.home_dist = {}
        for src in self.home_set:
            d = {src: 0}
            frontier = [src]
            while frontier:
                nxt = []
                for c in frontier:
                    for n in self.neighbors[c]:
                        if n in self.home_set and n not in d:
                            d[n] = d[c] + 1
                            nxt.append(n)
                frontier = nxt
            self.home_dist[src] = d

        # Opponent beliefs start exactly on their spawn points
        self.enemy_start = {}
        self.beliefs = {}
        self.prev_enemy = {}
        for i in self.get_opponents(game_state):
            start = self._cell(game_state.get_initial_agent_position(i))
            self.enemy_start[i] = start
            b = util.Counter()
            b[start] = 1.0
            self.beliefs[i] = b
            self.prev_enemy[i] = self._snapshot(game_state.get_agent_state(i))

        # Attacker / camper bookkeeping (fixes the face-off at the border)
        opps = self.get_opponents(game_state)
        self.pacman_turns = {i: 0 for i in opps}   # time spent on our side
        self.loiter = {i: 0 for i in opps}         # consecutive turns parked at border
        self.hover = {i: 0 for i in opps}          # consecutive turns hovering near it
        self.turn = 0
        self.camper = {i: False for i in opps}     # parks at the border, never enters
        self.last_action = None

        # Map intelligence: dead ends are pockets/corridors with a single way out.
        # A guard standing in one sees nothing and needs extra steps to react.
        self.dead_ends = self._find_dead_ends()
        border_dist = {c: min(self.get_maze_distance(c, e) for e in self.home_border)
                       for c in self.home_cells}
        self.guard_cells = [c for c in self.home_cells
                            if c not in self.dead_ends
                            and self.AMBUSH_DEPTH <= border_dist[c] <= self.GUARD_BAND]
        if not self.guard_cells:  # tiny or odd maps
            self.guard_cells = [c for c in self.home_cells if c not in self.dead_ends] \
                or list(self.home_cells)

        self.prev_food = set(self.get_food_you_are_defending(game_state).as_list())
        self.prev_capsules = set(self.get_capsules_you_are_defending(game_state))
        self.last_eaten = None
        self.hotspot_cache = (None, None)

    # ------------------------------------------------------------------ #
    # Main decision                                                       #
    # ------------------------------------------------------------------ #
    def choose_action(self, game_state):
        try:
            self._update_beliefs(game_state)
            raid = self._raid_action(game_state)
            if raid is not None:
                return raid
            target, threats, scared = self._choose_target(game_state)
            return self._best_action_towards(game_state, target, threats, scared)
        except Exception as e:  # never crash in the contest
            if self.DEBUG:
                import traceback
                traceback.print_exc()
            return super().choose_action(game_state)

    # ------------------------------------------------------------------ #
    # 1. KNOW: belief tracking                                            #
    # ------------------------------------------------------------------ #
    def _update_beliefs(self, game_state):
        my_pos = self._cell(game_state.get_agent_position(self.index))
        team_pos = [self._cell(game_state.get_agent_position(t))
                    for t in self.get_team(game_state)
                    if game_state.get_agent_position(t) is not None]

        try:
            noisy = game_state.get_agent_distances()
        except Exception:
            noisy = None

        opponents = self.get_opponents(game_state)
        states = {i: game_state.get_agent_state(i) for i in opponents}

        # What disappeared from our side since our last turn?
        food_now = set(self.get_food_you_are_defending(game_state).as_list())
        caps_now = set(self.get_capsules_you_are_defending(game_state))
        eaten_food = self.prev_food - food_now
        eaten_caps = self.prev_capsules - caps_now
        if eaten_food or eaten_caps:
            self.last_eaten = next(iter(eaten_food or eaten_caps))

        # Pass 1: time elapses (each enemy moved at most once)
        elapsed = {i: self._elapse(self.beliefs[i]) for i in opponents}

        # Pass 2: who ate what?  num_carrying going up identifies the eater
        pinned = {}
        for cell in eaten_food:
            cands = [i for i in opponents
                     if states[i].num_carrying > self.prev_enemy[i]['num_carrying']
                     and i not in pinned]
            if not cands:
                cands = [i for i in opponents if states[i].is_pacman and i not in pinned]
            if cands:
                pinned[max(cands, key=lambda i: elapsed[i][cell])] = cell
        for cell in eaten_caps:
            cands = [i for i in opponents if states[i].is_pacman and i not in pinned]
            if cands:
                pinned[max(cands, key=lambda i: elapsed[i][cell])] = cell

        # Pass 3: apply all evidence
        for i in opponents:
            st, prev = states[i], self.prev_enemy[i]
            pos = st.get_position()

            if pos is not None:                       # directly visible
                b = util.Counter()
                b[self._cell(pos)] = 1.0
            elif i in pinned:                         # caught eating our food
                b = util.Counter()
                b[pinned[i]] = 1.0
            else:
                b = elapsed[i]
                allowed = None

                if prev['is_pacman'] and not st.is_pacman:
                    # Left our side: either went home (border) or got eaten (spawn)
                    returned = st.num_returned > prev['num_returned']
                    died = prev['num_carrying'] > 0 and not returned
                    border_b = util.Counter({c: b[c] for c in self.enemy_border if b[c] > 0})
                    if died:
                        b = util.Counter({self.enemy_start[i]: 1.0})
                    else:
                        b = self._normalized(border_b)
                        if not returned:  # carried nothing: could also be dead
                            for c in list(b.keys()):
                                b[c] *= 0.5
                            b[self.enemy_start[i]] += 0.5
                elif not prev['is_pacman'] and st.is_pacman:
                    # Just crossed into our half -> must be on our border column
                    allowed = set(self.home_border)

                b = self._apply_constraints(b, st.is_pacman, team_pos, my_pos,
                                            noisy[i] if noisy else None, allowed)
            self.beliefs[i] = self._normalized(b)
            self.prev_enemy[i] = self._snapshot(st)
            self._update_role(game_state, i, st)

        self.prev_food = food_now
        self.prev_capsules = caps_now

    def _update_role(self, game_state, i, st):
        """Tell real attackers apart from ghosts that only camp or hover near the border."""
        if i == self.get_opponents(game_state)[0]:
            self.turn += 1
        if st.is_pacman:
            self.pacman_turns[i] += 1
            self.loiter[i] = self.hover[i] = 0
            self.camper[i] = False          # it came in: it is an attacker now
            return
        pos, _ = self._estimate(game_state, i)
        d_border = min(self.get_maze_distance(pos, e) for e in self.home_border)
        self.loiter[i] = self.loiter[i] + 1 if d_border <= 2 else 0
        self.hover[i] = self.hover[i] + 1 if d_border <= self.SHADOW_TRIGGER else 0
        mate_attacks = any(self.pacman_turns[j] > 0 for j in self.pacman_turns if j != i)
        if self.loiter[i] >= self.CAMP_LIMIT or self.hover[i] >= self.HOVER_LIMIT or \
                (mate_attacks and self.pacman_turns[i] == 0 and self.turn >= self.DEFENDER_AFTER):
            self.camper[i] = True           # sticky until it actually enters

    def _apply_constraints(self, b, is_pacman, team_pos, my_pos, noisy_dist, allowed):
        def side_ok(c):
            return self._is_home(c) == is_pacman

        def hidden_ok(c):
            return all(self._manhattan(c, t) > self.SIGHT_RANGE for t in team_pos)

        def sonar_ok(c):
            return noisy_dist is None or abs(self._manhattan(c, my_pos) - noisy_dist) <= self.SONAR_NOISE

        def allowed_ok(c):
            return allowed is None or c in allowed

        # Hard constraints first; sonar is applied only if it leaves something
        hard = util.Counter({c: w for c, w in b.items()
                             if w > 0 and side_ok(c) and hidden_ok(c) and allowed_ok(c)})
        with_sonar = util.Counter({c: w for c, w in hard.items() if sonar_ok(c)})
        if with_sonar and sum(with_sonar.values()) > 0:
            return with_sonar
        if hard and sum(hard.values()) > 0:
            return hard

        # Belief collapsed (we lost track): restart uniformly over consistent cells
        for check_sonar in (True, False):
            cells = [c for c in self.legal_cells
                     if side_ok(c) and hidden_ok(c) and allowed_ok(c)
                     and (not check_sonar or sonar_ok(c))]
            if cells:
                return util.Counter({c: 1.0 for c in cells})
        cells = [c for c in self.legal_cells if side_ok(c)] or self.legal_cells
        return util.Counter({c: 1.0 for c in cells})

    def _elapse(self, b):
        nb = util.Counter()
        for p, w in b.items():
            if w <= 0:
                continue
            nbrs = self.neighbors.get(p, [p])
            share = w / len(nbrs)
            for n in nbrs:
                nb[n] += share
        return nb

    def _estimate(self, game_state, i):
        pos = game_state.get_agent_state(i).get_position()
        if pos is not None:
            return self._cell(pos), True
        b = self.beliefs.get(i)
        if not b:
            return self.enemy_start[i], False
        return max(b.items(), key=lambda kv: kv[1])[0], False

    # ------------------------------------------------------------------ #
    # 2. DECIDE: where should we stand?                                   #
    # ------------------------------------------------------------------ #
    def _choose_target(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = self._cell(my_state.get_position())
        scared = my_state.scared_timer > 1   # timer still >0 after our move
        opponents = self.get_opponents(game_state)

        invaders = []
        for i in opponents:
            st = game_state.get_agent_state(i)
            if st.is_pacman:
                pos, visible = self._estimate(game_state, i)
                invaders.append((i, pos, visible, st.num_carrying))
        threats = [pos for (_, pos, visible, _) in invaders if visible]

        # ---- HUNT mode -------------------------------------------------
        if invaders:
            def priority(inv):
                _, pos, visible, carrying = inv
                return 3 * carrying - 0.5 * self._hd(my_pos, pos) + (5 if visible else 0)

            _, inv_pos, _, carrying = max(invaders, key=priority)
            d_me = self._hd(my_pos, inv_pos)

            # Block a capsule the invader is about to eat (it would scare us)
            if not scared and d_me > 2:
                for cap in self.get_capsules_you_are_defending(game_state):
                    d_inv_cap = self.get_maze_distance(inv_pos, cap)
                    if d_inv_cap <= 4 and self._hd(my_pos, cap) <= d_inv_cap:
                        return cap, threats, scared

            return self._intercept(game_state, my_pos, inv_pos, carrying), threats, scared

        # ---- GUARD mode ------------------------------------------------
        # Mirror only a ghost that behaves like an attacker. Campers (e.g. the
        # enemy defender parked on its own border) are ignored, otherwise the
        # two ghosts lock face to face in the middle forever.
        best_ghost, best_key = None, None
        for i in opponents:
            if self.camper[i]:
                continue
            pos, _ = self._estimate(game_state, i)
            d = min(self.get_maze_distance(pos, e) for e in self.home_border)
            if d > self.SHADOW_TRIGGER:
                continue
            key = (-self.pacman_turns[i], d)   # proven attackers first, then closest
            if best_key is None or key < best_key:
                best_ghost, best_key = pos, key

        if best_ghost is not None:
            attackers = [best_ghost]           # one real attacker is coming
        else:                                  # nobody close: cover every enemy
            attackers = [self._estimate(game_state, i)[0] for i in opponents]
        return self._goalkeeper(game_state, attackers, my_pos), threats, scared

    def _intercept(self, game_state, my_pos, inv_pos, carrying):
        """Meet the invader on the path it is about to take, not where it is now."""
        if self._hd(my_pos, inv_pos) <= 2:
            return inv_pos

        food = self.get_food_you_are_defending(game_state).as_list()
        try:
            time_left = game_state.data.timeleft
        except Exception:
            time_left = 9999

        heading_home = carrying >= self.CARRY_ESCAPE or not food or \
            (carrying > 0 and time_left < 80)
        goals = self.home_border if heading_home else food
        goal = min(goals, key=lambda g: self.get_maze_distance(inv_pos, g))
        total = self.get_maze_distance(inv_pos, goal)

        best, best_key = None, None
        for c in self.home_cells:
            d_inv = self.get_maze_distance(inv_pos, c)
            if d_inv + self.get_maze_distance(c, goal) != total:
                continue                          # not on its shortest path
            d_me = self._hd(my_pos, c)
            if d_me > d_inv:
                continue                          # we would arrive too late
            key = (d_inv, d_me)                   # earliest meeting point first
            if best_key is None or key < best_key:
                best, best_key = c, key
        return best if best is not None else inv_pos

    def _goalkeeper(self, game_state, attackers, my_pos):
        """
        Minimax guard spot. For every candidate cell c and every pellet/capsule f
        we protect, margin = d(c, f) - d(attacker, f): how much later than the
        attacker we would reach f.  We stand where the WORST margin is smallest,
        i.e. the attacker cannot reach any of our food much before we do,
        whichever entry it picks.  Candidates exclude dead ends and stay
        AMBUSH_DEPTH..GUARD_BAND steps inside, so we never block the line
        face to face and never sit in a pocket.
        """
        targets = self.get_food_you_are_defending(game_state).as_list() + \
            list(self.get_capsules_you_are_defending(game_state))
        if not targets:
            return self._food_hotspot(game_state)
        dist = self.get_maze_distance
        enemy_d = {f: min(dist(a, f) for a in attackers) for f in targets}
        best, best_key = None, None
        for c in self.guard_cells:
            worst = max(self._hd(c, f) - enemy_d[f] for f in targets)
            key = (worst, self._hd(my_pos, c))  # tie: the spot closest to us
            if best_key is None or key < best_key:
                best, best_key = c, key
        return best

    def _find_dead_ends(self):
        """Cells inside dead-end pockets, found by peeling off 1-exit cells."""
        deg = {c: len(self.neighbors[c]) - 1 for c in self.legal_cells}
        stack = [c for c in self.legal_cells if deg[c] <= 1]
        dead = set()
        while stack:
            c = stack.pop()
            if c in dead:
                continue
            dead.add(c)
            for n in self.neighbors[c]:
                if n != c and n not in dead:
                    deg[n] -= 1
                    if deg[n] <= 1:
                        stack.append(n)
        return dead

    def _food_hotspot(self, game_state):
        """Border cell that sits closest to the bulk of our remaining food."""
        food = self.get_food_you_are_defending(game_state).as_list()
        key = len(food)
        if self.hotspot_cache[0] == key and self.hotspot_cache[1] is not None:
            return self.hotspot_cache[1]
        cells = self.guard_cells
        if not food:
            spot = cells[len(cells) // 2]
        else:
            spot = max(cells,
                       key=lambda e: sum(1.0 / (1 + self.get_maze_distance(e, f)) for f in food))
        self.hotspot_cache = (key, spot)
        return spot

    # ------------------------------------------------------------------ #
    # 3. ACT + 4. SAFETY: pick the move                                   #
    # ------------------------------------------------------------------ #
    def _best_action_towards(self, game_state, target, threats, scared):
        actions = game_state.get_legal_actions(self.index)
        best_score, best_actions = None, []

        for a in actions:
            succ = self.get_successor(game_state, a)
            s_state = succ.get_agent_state(self.index)
            s_pos = self._cell(s_state.get_position())

            score = -10 * self._hd(s_pos, target)
            if s_state.is_pacman:
                score -= 10000                    # a defender stays home
            if a == Directions.STOP:
                score -= 1                        # prefer moving when it is free

            for t in threats:
                d = self.get_maze_distance(s_pos, t)
                if scared and d <= 1:
                    score -= 5000                 # it can eat us next turn
                elif not scared and d == 0:
                    score += 5000                 # we eat it now

            if best_score is None or score > best_score:
                best_score, best_actions = score, [a]
            elif score == best_score:
                best_actions.append(a)

        # Stable tie-break: keep going the same way instead of jittering
        if self.last_action in best_actions:
            choice = self.last_action
        else:
            moving = [a for a in best_actions if a != Directions.STOP]
            choice = random.choice(moving or best_actions)
        self.last_action = choice
        return choice

    # ------------------------------------------------------------------ #
    # 5. SCORE: safe hit-and-run raids                                    #
    # ------------------------------------------------------------------ #
    # A perfect defence only gives 0-0 ties; one pellet brought home wins.
    # When nobody is attacking us, the defender slips over the border, eats
    # a pellet that it can provably reach and bring back before any enemy
    # ghost can interfere, and returns. Any invader cancels the raid.
    def _raid_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = self._cell(my_state.get_position())
        opponents = self.get_opponents(game_state)
        invaders = [i for i in opponents if game_state.get_agent_state(i).is_pacman]

        if not my_state.is_pacman:
            if not self.RAID or invaders or my_state.scared_timer > 0:
                return None
            # An attacker lining up at our border needs the goalkeeper
            for i in opponents:
                if self.camper[i]:
                    continue
                pos, _ = self._estimate(game_state, i)
                if min(self.get_maze_distance(pos, e) for e in self.home_border) <= self.SHADOW_TRIGGER:
                    return None
            target = self._raid_target(game_state, my_pos, my_state.num_carrying)
            if target is None:
                return None
            return self._raid_move(game_state, target, returning=False)

        # Already on the enemy side: continue, or run home
        carry_goal = self.RAID_CARRY
        if all(game_state.get_agent_state(i).scared_timer > self.RAID_TRIP for i in opponents):
            carry_goal = 2 * self.RAID_CARRY + 2     # their ghosts are harmless
        danger = self._ghost_danger(game_state, my_pos)
        try:
            time_left = game_state.data.timeleft
        except Exception:
            time_left = 9999
        home_d = min(self.get_maze_distance(my_pos, e) for e in self.home_border)
        go_home = invaders or my_state.num_carrying >= carry_goal or danger <= 4 \
            or time_left <= 4 * (home_d + 4)
        if not go_home:
            target = self._raid_target(game_state, my_pos, my_state.num_carrying)
            if target is not None:
                return self._raid_move(game_state, target, returning=False)
        home = min(self.home_border, key=lambda e: self.get_maze_distance(my_pos, e))
        return self._raid_move(game_state, home, returning=True)

    def _raid_target(self, game_state, my_pos, carrying):
        food = self.get_food(game_state).as_list()
        if not food:
            return None
        dist = self.get_maze_distance
        best, best_trip = None, None
        for f in food:
            d_me = dist(my_pos, f)
            trip = d_me + min(dist(f, e) for e in self.home_border)
            if trip > self.RAID_TRIP or (best_trip is not None and trip >= best_trip):
                continue
            if self._raid_safe(game_state, f, d_me, trip):
                best, best_trip = f, trip
        return best

    def _raid_safe(self, game_state, f, d_me, trip):
        """No dangerous ghost can plausibly reach f within d_me + RAID_SAFETY."""
        reach = d_me + self.RAID_SAFETY
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            if st.is_pacman or st.scared_timer > trip + 2:
                continue                       # not on guard / harmless for the trip
            pos = st.get_position()
            if pos is not None:
                if self.get_maze_distance(self._cell(pos), f) <= reach:
                    return False
                continue
            risk = sum(w for c, w in self.beliefs[i].items()
                       if self.get_maze_distance(c, f) <= reach)
            if risk > self.RAID_RISK:
                return False
        return True

    def _ghost_danger(self, game_state, my_pos):
        d = 99
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            pos = st.get_position()
            if pos is None or st.is_pacman or st.scared_timer > 2:
                continue
            d = min(d, self.get_maze_distance(my_pos, self._cell(pos)))
        return d

    def _raid_move(self, game_state, target, returning):
        ghosts = []
        for i in self.get_opponents(game_state):
            st = game_state.get_agent_state(i)
            if st.get_position() is not None and not st.is_pacman and st.scared_timer <= 2:
                ghosts.append(self._cell(st.get_position()))
        best, best_score = [], None
        for a in game_state.get_legal_actions(self.index):
            succ = self.get_successor(game_state, a)
            s_state = succ.get_agent_state(self.index)
            s_pos = self._cell(s_state.get_position())
            score = -10 * self.get_maze_distance(s_pos, target)
            if returning and not s_state.is_pacman:
                score += 1000                  # safely home
            if s_pos == self.start:
                score -= 100000                # that move gets us eaten
            for g in ghosts:
                d = self.get_maze_distance(s_pos, g)
                if s_state.is_pacman and d <= 1:
                    score -= 50000
                elif s_state.is_pacman and d == 2:
                    score -= 500
            if a == Directions.STOP:
                score -= 5
            if best_score is None or score > best_score:
                best, best_score = [a], score
            elif score == best_score:
                best.append(a)
        choice = random.choice(best)
        self.last_action = choice
        return choice

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _is_home(self, cell):
        return cell[0] < self.mid_x if self.red else cell[0] >= self.mid_x

    @staticmethod
    def _cell(pos):
        p = nearest_point(pos)
        return int(p[0]), int(p[1])

    def _hd(self, a, b):
        """Our distance from a to b while staying on our side."""
        row = self.home_dist.get(a)
        if row is not None and b in row:
            return row[b]
        return self.get_maze_distance(a, b) + 50   # off our side or cut off: discourage

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _snapshot(st):
        return {'is_pacman': st.is_pacman,
                'num_carrying': st.num_carrying,
                'num_returned': st.num_returned}

    @staticmethod
    def _normalized(b):
        total = sum(b.values())
        if total <= 0:
            return b
        out = util.Counter()
        for k, v in b.items():
            if v > 0:
                out[k] = v / total
        return out

    # Original reflex features kept as the fallback used by choose_action
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
