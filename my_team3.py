# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
#
# ---------------------------------------------------------------------------
# Pacman Capture-the-Flag agent -- Autonomous Systems, EMAI 2026-27 (UPF)
#
# The whole agent lives in this ONE file on purpose: the contest loader adds
# our folder to sys.path, so if we split code into e.g. "utils.py" and the
# opponent team also has a "utils.py", Python's module cache could give one
# team the other team's file. One file = no name clashes on the cluster.
#
# The file follows our team diagram:
#
#     env data + agent data --> EVALUATOR --(role, objective)--> EXECUTOR
#                                                (offensive / defensive)
#                                                  |  actions   ^ correction
#                                                  v            |
#                                                     JUDGE ----+
#
#   SECTION 1  Map Intelligence   (the Field Map)   - distances, dead ends,
#                                                     chokepoints, Voronoi
#   SECTION 1b Enemy Tracker                        - where are the enemies
#                                                     we cannot see?
#   SECTION 2  EVALUATOR          (the Coach)       - role AND objective on the
#                                                     map for each agent (Plan)
#   SECTION 3  JUDGE                                - scores board states for the
#                                                     search, and reviews every
#                                                     move, sending corrections
#   SECTION 4  EXECUTOR           (the Search)      - A* for travel, alpha-beta
#                                                     for close combat; offensive
#                                                     and defensive modules are
#                                                     in CoachedAgent
#   SECTION 5  Safety Net         (Guardrails)      - time budget, never crash
#
# Python 3.8 compatible (the cluster's version): no 3.9+ syntax is used.
# ---------------------------------------------------------------------------

import heapq
import os
import random
import time
from collections import deque

from contest.capture_agents import CaptureAgent
from contest.game import Directions

# Movement vectors for the four real moves (STOP keeps you in place).
MOVES = {
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST: (1, 0),
    Directions.WEST: (-1, 0),
}

SIGHT_RANGE = 5          # opponents are visible within this Manhattan distance
SONAR_NOISE = 6          # noisy distance = true Manhattan distance +/- 6

# --- Time budget (SECTION 5) -----------------------------------------------
# The rule is 1 s per move (3 warnings = forfeit, one move > 3 s = forfeit).
# The cluster runs many games at once and can be slower than a laptop, so we
# only ever use a fraction of that second.
SEARCH_TIME_BUDGET = 0.30   # seconds allowed for the alpha-beta search
HARD_TIME_LIMIT = 0.75      # if a move ever gets near this, we stop thinking

INF = float('inf')

# --- Experiment switches -----------------------------------------------------
# Ideas that were tested and did NOT win more than the version without them
# (40 head-to-head games). Kept here, switched off, so the team can re-test
# them against stronger opponents after the Nov 23 test run.
USE_EXIT_INTERCEPT = False   # defender waits at the exit of a loaded HIDDEN invader
USE_HEAVY_RECALL = False     # attacker comes back when one invader carries >= 5
USE_BAIT = True              # defender opens the door to a waiting attacker, then eats it

# Set PACMAN_DEBUG=1 to let errors crash loudly and print slow moves (local testing only).
DEBUG = os.environ.get('PACMAN_DEBUG') == '1'


#############################################################################
# Team creation (required by the framework)                                 #
#############################################################################

def create_team(first_index, second_index, is_red,
                first='CoachedAgent', second='CoachedAgent', num_training=0):
    """Both agents are the same class; the Coach gives them different jobs."""
    return [eval(first)(first_index), eval(second)(second_index)]


#############################################################################
# SECTION 1 - MAP INTELLIGENCE (the Field Map)                              #
#############################################################################

class MapIntel:
    """
    Turns the raw wall grid into tactical knowledge. Computed ONCE per map
    (inside the 15-second start-up allowance) and shared by both agents.

      dist[a][b]   true walking distance between any two open cells
      home_border  our cells touching the middle line (where we cross)
      trap_depth   for cells inside dead ends: how many steps to get out
      trap_exit    the cell where that dead end opens up again
      chokepoints  articulation points: cells whose removal splits the maze
    """

    _cache = {}

    @classmethod
    def get(cls, game_state, red):
        walls = game_state.get_walls()
        key = (tuple(tuple(col) for col in walls.data), red)
        if key not in cls._cache:
            cls._cache[key] = MapIntel(walls, red)
        return cls._cache[key]

    def __init__(self, walls, red):
        self.red = red
        self.width = walls.width
        self.height = walls.height

        # ---- 1. Open cells and their neighbours ---------------------------
        self.cells = [(x, y) for x in range(self.width)
                      for y in range(self.height) if not walls[x][y]]
        cell_set = set(self.cells)
        self.neighbors = {}      # cell -> list of neighbouring open cells
        for (x, y) in self.cells:
            nbrs = []
            for dx, dy in MOVES.values():
                n = (x + dx, y + dy)
                if n in cell_set:
                    nbrs.append(n)
            self.neighbors[(x, y)] = nbrs
        # neighbours plus "stay where you are", used by the search
        self.moves_from = dict((c, self.neighbors[c] + [c]) for c in self.cells)

        # ---- 2. All-pairs walking distances (one BFS per cell) -------------
        self.dist = {}
        for src in self.cells:
            d = {src: 0}
            q = deque([src])
            while q:
                c = q.popleft()
                for n in self.neighbors[c]:
                    if n not in d:
                        d[n] = d[c] + 1
                        q.append(n)
            self.dist[src] = d

        # ---- 3. Home side and border cells ---------------------------------
        # Red owns x < width/2, blue owns x >= width/2.
        mid = self.width // 2
        self.home_x = mid - 1 if red else mid
        self.enemy_x = mid if red else mid - 1
        self.home_border = [(self.home_x, y) for y in range(self.height)
                            if (self.home_x, y) in cell_set
                            and (self.enemy_x, y) in cell_set]
        self.enemy_border = [(self.enemy_x, y) for (_, y) in self.home_border]

        # distance from every cell to the nearest home-border cell
        self.dist_home = {}
        for c in self.cells:
            self.dist_home[c] = min(self.dist[c].get(b, 9999) for b in self.home_border) \
                if self.home_border else 0

        # distances using ONLY cells of our own half (how a defender that
        # must not cross the line actually walks)
        home_cells = [c for c in self.cells if self.is_home(c)]
        self.home_dist = {}
        for src in home_cells:
            d = {src: 0}
            q = deque([src])
            while q:
                c = q.popleft()
                for n in self.neighbors[c]:
                    if n not in d and self.is_home(n):
                        d[n] = d[c] + 1
                        q.append(n)
            self.home_dist[src] = d

        # ---- 4. Dead ends ("traps") -----------------------------------------
        # Repeatedly peel off cells with only one open neighbour. Whatever gets
        # peeled lies in a dead-end tree: walk in and the only way out is back.
        degree = dict((c, len(self.neighbors[c])) for c in self.cells)
        removed = set()
        q = deque([c for c in self.cells if degree[c] <= 1])
        while q:
            c = q.popleft()
            if c in removed:
                continue
            removed.add(c)
            for n in self.neighbors[c]:
                if n not in removed:
                    degree[n] -= 1
                    if degree[n] == 1:
                        q.append(n)
        # Multi-source BFS from the "core" (non-dead-end cells) gives depth
        # and the exit cell of every dead-end cell.
        self.trap_depth = dict((c, 0) for c in self.cells)
        self.trap_exit = dict((c, c) for c in self.cells)
        core = [c for c in self.cells if c not in removed]
        q = deque(core)
        seen = set(core)
        while q:
            c = q.popleft()
            for n in self.neighbors[c]:
                if n not in seen and n in removed:
                    seen.add(n)
                    self.trap_depth[n] = self.trap_depth[c] + 1
                    self.trap_exit[n] = self.trap_exit[c]
                    q.append(n)

        # ---- 5. Chokepoints (articulation points, iterative Tarjan) ---------
        self.chokepoints = self._articulation_points()

    def _articulation_points(self):
        disc, low, parent = {}, {}, {}
        points = set()
        timer = [0]
        for root in self.cells:
            if root in disc:
                continue
            parent[root] = None
            disc[root] = low[root] = timer[0]
            timer[0] += 1
            children = 0
            stack = [(root, iter(self.neighbors[root]))]
            while stack:
                node, it = stack[-1]
                advanced = False
                for n in it:
                    if n not in disc:
                        parent[n] = node
                        disc[n] = low[n] = timer[0]
                        timer[0] += 1
                        if node == root:
                            children += 1
                        stack.append((n, iter(self.neighbors[n])))
                        advanced = True
                        break
                    elif n != parent[node]:
                        low[node] = min(low[node], disc[n])
                if not advanced:
                    stack.pop()
                    if stack:
                        p = stack[-1][0]
                        low[p] = min(low[p], low[node])
                        if parent[p] is not None and low[node] >= disc[p]:
                            points.add(p)
            if children > 1:
                points.add(root)
        return points

    # ---- helpers used everywhere -------------------------------------------
    def d(self, a, b):
        """Walking distance (9999 if unreachable / not a grid cell)."""
        return self.dist.get(a, {}).get(b, 9999)

    def hd(self, a, b):
        """Walking distance staying inside our half (falls back to normal distance + penalty)."""
        d = self.home_dist.get(a, {}).get(b)
        if d is None:
            return self.d(a, b) + 50
        return d

    def is_home(self, cell):
        return cell[0] < self.width // 2 if self.red else cell[0] >= self.width // 2

    def safe_border_cells(self, me, threats):
        """
        Voronoi idea: the home-border cells I reach STRICTLY before every
        threatening ghost. If this list is empty I can be cut off.
        """
        out = []
        for b in self.home_border:
            mine = self.d(me, b)
            if all(mine < self.d(g, b) for g in threats):
                out.append(b)
        return out


#############################################################################
# SECTION 1b - ENEMY TRACKER (where are the enemies we cannot see?)         #
#############################################################################

class EnemyTracker:
    """
    Keeps a probability map over cells for each opponent (exact Bayesian
    filter, like the "ghostbusters" project). Shared by both teammates, so
    the readings of both agents are combined.

      elapse   : an enemy that just moved can be on any neighbour cell
      observe  : (1) read the enemy's state fields, which stay visible even
                     when its position is hidden (see the professor's forum
                     post, 27 Oct): is_pacman, num_carrying, num_returned,
                     scared_timer. Changes in them reveal exact events:
                       num_carrying +1  and a pellet gone -> it is ON that cell
                       num_returned up                    -> it just crossed home
                       num_carrying -> 0, nothing banked  -> it was eaten (start)
                       scared_timer reset to 0            -> it was eaten (start)
                 (2) otherwise filter: noisy distance, cells we would see,
                     which half it is on (is_pacman).

    We always compare with the LAST state seen by EITHER of our agents. Turn
    order alternates (us, them, us, them), so since that state exactly one
    enemy has moved: every change can be pinned on the right enemy.
    """

    def __init__(self, mi, game_state, enemies):
        self.mi = mi
        self.enemies = enemies
        self.belief = {}
        self.last_state = None        # last observation of either teammate
        self.last_mover = None        # the enemy that moved just before this look
        self.capsule_cells = set()    # our capsules, to tell capsules from pellets
        for e in enemies:
            self.reset_to_start(e, game_state)

    def reset_to_start(self, e, game_state):
        self.belief[e] = {game_state.get_initial_agent_position(e): 1.0}

    def elapse(self, e):
        new = {}
        for c, p in self.belief[e].items():
            options = self.mi.moves_from[c]
            share = p / len(options)
            for n in options:
                new[n] = new.get(n, 0.0) + share
        self.belief[e] = new

    def observe(self, agent, game_state):
        mi = self.mi
        prev = self.last_state
        self.last_mover = (agent.index - 1) % game_state.get_num_agents()
        if prev is not None:
            self.capsule_cells = set(agent.get_capsules_you_are_defending(prev))
        my_pos = game_state.get_agent_position(agent.index)
        team_pos = [game_state.get_agent_position(i) for i in agent.get_team(game_state)]
        noisy = game_state.get_agent_distances()

        # Pellets / capsules that vanished from OUR side since the last look.
        vanished = []
        if prev is not None:
            before = agent.get_food_you_are_defending(prev).as_list()
            now = set(agent.get_food_you_are_defending(game_state).as_list())
            vanished = [f for f in before if f not in now]
            caps_now = set(agent.get_capsules_you_are_defending(game_state))
            vanished += [c for c in agent.get_capsules_you_are_defending(prev) if c not in caps_now]

        for e in self.enemies:
            st = game_state.get_agent_state(e)
            pos = game_state.get_agent_position(e)
            if pos is not None:                       # we can see it
                self.belief[e] = {pos: 1.0}
                continue

            if prev is not None and self._read_state_change(e, prev.get_agent_state(e), st,
                                                            vanished, game_state):
                continue                              # pinned exactly, no filtering needed

            new = {}
            for c, p in self.belief[e].items():
                if mi.is_home(c) != st.is_pacman:     # wrong half for its form
                    continue
                if any(abs(c[0] - t[0]) + abs(c[1] - t[1]) <= SIGHT_RANGE for t in team_pos):
                    continue                          # we'd see it there
                if noisy is not None:
                    md = abs(c[0] - my_pos[0]) + abs(c[1] - my_pos[1])
                    if abs(noisy[e] - md) > SONAR_NOISE:
                        continue
                new[c] = p
            if not new:
                new = self._fresh_belief(e, st, team_pos, my_pos, noisy)
            total = sum(new.values())
            self.belief[e] = dict((c, p / total) for c, p in new.items())

        self.last_state = game_state

    def _read_state_change(self, e, ps, st, vanished, game_state):
        """
        Uses the enemy's always-visible fields. Returns True if the enemy's
        position is now known exactly (or reset), so filtering can be skipped.
        """
        mi = self.mi
        # 1) It banked food: it is standing on its own side of the border.
        if st.num_returned > ps.num_returned:
            prior = self.belief[e]
            new = {}
            for b in mi.enemy_border:
                w = sum(prior.get(n, 0.0) for n in mi.moves_from[b])
                new[b] = w
            if sum(new.values()) == 0:
                new = dict((b, 1.0) for b in mi.enemy_border)
            total = sum(new.values())
            if total > 0:
                self.belief[e] = dict((c, p / total) for c, p in new.items())
                return True
            return False
        # 2) Its load vanished without being banked: it was eaten -> start.
        if ps.num_carrying > 0 and st.num_carrying == 0:
            self.reset_to_start(e, game_state)
            return True
        # 3) A scared ghost's timer reset to 0 early: it was eaten -> start.
        if ps.scared_timer > 1 and st.scared_timer == 0 and not st.is_pacman:
            self.reset_to_start(e, game_state)
            return True
        # 4) It ate a pellet: it is standing on the pellet that vanished.
        if st.num_carrying > ps.num_carrying and vanished:
            if len(vanished) == 1:
                self.belief[e] = {vanished[0]: 1.0}
            else:
                prior = self.belief[e]
                spot = max(vanished, key=lambda f: sum(prior.get(n, 0.0) for n in mi.moves_from[f]))
                self.belief[e] = {spot: 1.0}
            return True
        # 4b) One of our capsules vanished (eating a capsule does not change
        #     num_carrying): the invader that just moved is standing on it.
        caps = [c for c in vanished if c in self.capsule_cells]
        if caps and st.is_pacman and e == self.last_mover:
            self.belief[e] = {caps[0]: 1.0}
            return True
        # 5) It was a Pacman deep in our half and is a ghost again, carrying
        #    nothing, without being seen crossing: most likely eaten.
        if ps.is_pacman and not st.is_pacman:
            likely = self.most_likely(e)
            if likely is not None and mi.is_home(likely) and mi.dist_home.get(likely, 0) > 2:
                self.reset_to_start(e, game_state)
                return True
        return False

    def _fresh_belief(self, e, st, team_pos, my_pos, noisy):
        """We lost track: start again from every cell consistent with what we know."""
        mi = self.mi
        new = {}
        for c in mi.cells:
            if mi.is_home(c) != st.is_pacman:
                continue
            if any(abs(c[0] - t[0]) + abs(c[1] - t[1]) <= SIGHT_RANGE for t in team_pos):
                continue
            if noisy is not None:
                md = abs(c[0] - my_pos[0]) + abs(c[1] - my_pos[1])
                if abs(noisy[e] - md) > SONAR_NOISE:
                    continue
            new[c] = 1.0
        if not new:
            new = dict((c, 1.0) for c in mi.cells if mi.is_home(c) == st.is_pacman)
        if not new:
            new = dict((c, 1.0) for c in mi.cells)
        return new

    def most_likely(self, e):
        b = self.belief.get(e)
        if not b:
            return None
        return max(b.items(), key=lambda kv: kv[1])[0]


#############################################################################
# SECTION 2 - EVALUATOR (the Coach): roles + objectives                     #
#############################################################################

def enemy_scared_window(agent, game_state):
    """Moves left before the first enemy ghost stops being scared (0 if any is not scared)."""
    timers = [game_state.get_agent_state(e).scared_timer
              for e in agent.get_opponents(game_state)
              if not game_state.get_agent_state(e).is_pacman]
    return min(timers) if timers else 0


ATTACK_ROLES = ('attack', 'raid', 'attack_return')


class Plan:
    """
    What the Evaluator hands to the Executor for ONE agent, every turn
    (the "role" and "objective in the map" arrows of the team diagram).

      role       attack / raid / attack_return / defend
      mode       the kind of objective, e.g. 'eat', 'return', 'capsule',
                 'entry', 'chase', 'pincer', 'guard_exit', 'hunt', 'patrol'
      objective  the main target cell (or None)
      goals      cells the executor may aim for (A* takes the nearest)
      info       extra facts the executor and the judge need (threats, ...)
    """

    def __init__(self, role, mode, objective=None, goals=None, **info):
        self.role = role
        self.mode = mode
        self.objective = objective
        if goals is None:
            goals = [objective] if objective is not None else []
        self.goals = goals
        self.info = info

    def __repr__(self):
        return 'Plan(%s, %s, %s)' % (self.role, self.mode, self.objective)


class TeamCoach:
    """
    One coach per team colour, shared by both agents. Every turn it looks at
    the whole board and hands out non-overlapping jobs:

      'attack'  - go eat enemy food, bring it home
      'raid'    - cautious attack (small loads, near the border) used by the
                  usual defender when the enemy is not attacking at all
      'defend'  - guard our half, chase invaders

    It also splits the enemy food when both agents are attacking, so they
    never chase the same pellet.
    """

    _coaches = {}

    @classmethod
    def for_team(cls, red):
        return cls._coaches.setdefault(red, TeamCoach())

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracker = None
        self.quiet_turns = 0          # our turns in a row with no invader
        self.roles = {}
        self.last_turn_seen = None

    def assign_roles(self, agent, game_state):
        team = sorted(agent.get_team(game_state))
        enemies = agent.get_opponents(game_state)
        mi = agent.mi
        score = agent.get_score(game_state)
        moves_left = game_state.data.timeleft // 4      # moves left per agent

        invaders = [e for e in enemies if game_state.get_agent_state(e).is_pacman]
        if invaders:
            self.quiet_turns = 0
        elif agent.index == team[0]:
            self.quiet_turns += 1          # counted once per round, not per agent

        attacker, defender = team[0], team[1]
        roles = {attacker: 'attack', defender: 'defend'}

        # (a) Protect a lead at the end of the game: everybody defends
        #     (an attacker still carrying food first brings it home).
        if score > 0 and moves_left < 50:
            roles = dict((i, 'defend') for i in team)

        # (b) Two invaders at once, or one invader carrying a big load
        #     (num_carrying is readable even when we can't see it): the
        #     attacker helps if it is not out on a run with food.
        elif (len(invaders) >= 2 and score >= -2) or (USE_HEAVY_RECALL and
                any(game_state.get_agent_state(e).num_carrying >= 5 for e in invaders)):
            a_state = game_state.get_agent_state(attacker)
            if a_state.num_carrying == 0 and not a_state.is_pacman:
                roles[attacker] = 'defend'

        # (c) The enemy has not attacked for a while: the defender joins
        #     the attack carefully (comes straight back if anyone invades).
        elif not invaders and self.quiet_turns > 40 and score < 6:
            roles[defender] = 'raid'

        # (d) Their ghosts are scared for a long time (scared_timer is
        #     readable even when unseen) and nobody is invading: free food.
        elif not invaders and enemy_scared_window(agent, game_state) >= 20:
            roles[defender] = 'raid'

        self.roles = roles
        return roles

    def food_for(self, agent, game_state, food_list):
        """Split food by height when both of us are on the attack."""
        attackers = [i for i, r in self.roles.items() if r in ('attack', 'raid')]
        if len(attackers) < 2 or len(food_list) < 6:
            return food_list
        ys = sorted(f[1] for f in food_list)
        median = ys[len(ys) // 2]
        other = [i for i in attackers if i != agent.index][0]
        my_y = game_state.get_agent_position(agent.index)[1]
        other_y = game_state.get_agent_position(other)[1]
        i_take_top = (my_y, agent.index) > (other_y, other)
        mine = [f for f in food_list if (f[1] >= median) == i_take_top]
        return mine if mine else food_list


    # ===================================================== EVALUATOR: objectives
    # The Coach is the EVALUATOR of the team diagram: it reads the environment
    # and agent data, and outputs a role AND an objective for each agent.

    def make_plan(self, agent, game_state):
        role = self.roles.get(agent.index, 'attack')
        me_state = game_state.get_agent_state(agent.index)
        # An attacker told to defend first banks what it is carrying.
        if role == 'defend' and me_state.is_pacman and me_state.num_carrying > 0:
            role = 'attack_return'
        if role in ATTACK_ROLES:
            return self._plan_attack(agent, game_state, role)
        return self._plan_defend(agent, game_state)

    def _plan_attack(self, agent, game_state, role):
        mi = agent.mi
        me = game_state.get_agent_position(agent.index)
        me_state = game_state.get_agent_state(agent.index)
        carried = me_state.num_carrying
        moves_left = game_state.data.timeleft // 4

        food_all = agent.get_food(game_state).as_list()
        food = self.food_for(agent, game_state, food_all)
        food = [f for f in food if agent.taboo.get(f, -1) < agent.turn] or food
        capsules = agent.get_capsules(game_state)

        # Visible ghosts that can hurt us (not scared, or about to wake up).
        threats = []
        for e in agent.get_opponents(game_state):
            st = game_state.get_agent_state(e)
            pos = game_state.get_agent_position(e)
            if pos is None or st.is_pacman:
                continue
            if st.scared_timer > mi.d(me, pos) + 2:
                continue            # scared long enough: harmless
            threats.append(pos)
        threats.sort(key=lambda g: mi.d(me, g))
        nearest_threat = mi.d(me, threats[0]) if threats else 9999

        # ---- Should we go home? ---------------------------------------------
        carry_limit = 2 if role == 'raid' else (4 if threats else 7)
        if enemy_scared_window(agent, game_state) > mi.dist_home[me] + 6:
            carry_limit = 12        # their ghosts stay scared long enough: feast
        nearest_food = min([mi.d(me, f) for f in food] or [9999])
        returning = carried > 0 and (
            role == 'attack_return'
            or len(food_all) <= 2
            or moves_left <= mi.dist_home[me] + 8
            or carried >= carry_limit
            or (carried >= 3 and nearest_food > mi.dist_home[me])
        )
        if len(food_all) <= 2:
            returning = True

        # Close combat is decided by the executor's alpha-beta search.
        fight = bool(threats) and nearest_threat <= 6 and me_state.is_pacman
        facts = dict(threats=threats, food=food, capsules=capsules,
                     returning=returning, carried=carried, fight=fight)

        if returning:
            goals = mi.safe_border_cells(me, threats) or mi.home_border
            if role == 'attack_return' or len(food_all) <= 2:
                goals = mi.home_border if not threats else goals
            return Plan(role, 'return', None, goals, **facts)

        # Capsule if a ghost is chasing us on ENEMY ground and we win the race
        # to it. (On our own half a ghost cannot hurt us, so this must not
        # override the "crossing is guarded, try another one" logic below.)
        if threats and nearest_threat <= 8 and me_state.is_pacman \
                and agent.turn >= agent.capsule_ban_until:
            good_caps = [c for c in capsules if mi.d(me, c) < min(mi.d(g, c) for g in threats)]
            if good_caps:
                return Plan(role, 'capsule', None, good_caps, **facts)

        # Stuck on our side because a ghost is guarding the crossing?
        if not me_state.is_pacman:
            agent.stuck_turns += 1 if threats else 0
            if agent.stuck_turns > 6:
                agent.entry_point = self.pick_entry(agent, me, threats, avoid_here=True)
                agent.stuck_turns = 0
            if agent.entry_point is not None:
                if me == agent.entry_point:
                    agent.entry_point = None
                else:
                    return Plan(role, 'entry', agent.entry_point, **facts)
        else:
            agent.stuck_turns = 0
            agent.entry_point = None

        target = self.pick_food(agent, me, food, threats, role)
        if target is None:
            return Plan(role, 'home', None, mi.home_border, **facts)
        return Plan(role, 'eat', target, **facts)

    def pick_food(self, agent, me, food, threats, role):
        """Closest food that is not a death trap; raiders prefer food near the border."""
        mi = agent.mi
        best, best_score = None, INF
        for f in food:
            df = mi.d(me, f)
            score = df
            # dead-end check: can a ghost reach the pocket exit before we get out?
            if threats and mi.trap_depth[f] > 0:
                ex = mi.trap_exit[f]
                my_out = df + mi.trap_depth[f]
                if min(mi.d(g, ex) for g in threats) <= my_out + 1:
                    score += 1000
            if threats:
                closest_g = min(mi.d(g, f) for g in threats)
                if closest_g <= df:
                    score += 50
            if role == 'raid':
                score += 1.5 * mi.dist_home[f]
            if score < best_score:
                best, best_score = f, score
        return best

    def pick_entry(self, agent, me, threats, avoid_here=False):
        """A crossing point far from the guarding ghost but not too far from us."""
        mi = agent.mi
        options = mi.home_border
        if avoid_here:
            # we are stuck where we stand: insist on a different crossing
            options = [b for b in mi.home_border if mi.d(me, b) >= 3] or mi.home_border
        if not options:
            return None

        def value(b):
            g_dist = min([mi.d(g, b) for g in threats] or [20])
            return g_dist - 0.5 * mi.d(me, b)
        return max(options, key=value)

    def _plan_defend(self, agent, game_state):
        mi = agent.mi
        me = game_state.get_agent_position(agent.index)
        scared = game_state.get_agent_state(agent.index).scared_timer > 0
        tracker = self.tracker

        visible_invaders = []      # positions
        hidden_invaders = []       # (likely position, food it carries)
        carrying_at = {}
        for e in agent.get_opponents(game_state):
            st = game_state.get_agent_state(e)
            if not st.is_pacman:
                continue
            pos = game_state.get_agent_position(e)
            if pos is not None:
                visible_invaders.append(pos)
                carrying_at[pos] = st.num_carrying
            else:
                guess = tracker.most_likely(e)
                if guess is not None:
                    hidden_invaders.append((guess, st.num_carrying))

        # Two defenders: the second one cuts off the escape route (pincer).
        defenders = sorted(i for i, r in self.roles.items() if r == 'defend')
        am_second = len(defenders) == 2 and agent.index == defenders[1]

        if visible_invaders:
            # the invader carrying the most food is the most urgent one
            inv = min(visible_invaders, key=lambda p: mi.hd(me, p) - 3 * carrying_at.get(p, 0))
            escape = min(mi.enemy_border, key=lambda b: mi.d(inv, b)) if mi.enemy_border else inv
            home_escape = (escape[0] - 1, escape[1]) if agent.red else (escape[0] + 1, escape[1])
            if home_escape not in mi.dist:
                home_escape = min(mi.home_border, key=lambda b: mi.d(b, escape))

            if scared:
                return Plan('defend', 'shadow', inv)       # can't eat it: keep distance 2
            if am_second and mi.hd(me, home_escape) <= mi.d(inv, home_escape):
                return Plan('defend', 'pincer', home_escape)
            # Trapped in a dead end? Guard the exit instead of following.
            if mi.trap_depth[inv] > 0:
                ex = mi.trap_exit[inv]
                if mi.is_home(ex) and mi.hd(me, ex) < mi.d(inv, ex):
                    if me == ex:
                        if mi.hd(me, inv) <= 1:
                            return Plan('defend', 'guard_exit', inv)
                        return Plan('defend', 'hold', me)
                    return Plan('defend', 'guard_exit', ex)
            return Plan('defend', 'chase', inv, home_escape=home_escape)

        if hidden_invaders:
            guess, load = max(hidden_invaders, key=lambda gl: (gl[1], -mi.hd(me, gl[0])))
            if USE_EXIT_INTERCEPT and load >= 3 and mi.home_border:
                exit_cell = min(mi.home_border, key=lambda b: mi.d(guess, b))
                return Plan('defend', 'intercept', exit_cell)
            return Plan('defend', 'hunt', guess)

        # Nobody inside. Is an enemy ghost waiting right across the line from
        # us (the border stand-off)? Then open the door and trap it.
        bait = self._bait_plan(agent, game_state, me, scared)
        if bait is not None:
            return bait

        # Otherwise patrol the border facing the most dangerous enemy.
        return Plan('defend', 'patrol', self.patrol_point(agent, game_state))

    def _bait_plan(self, agent, game_state, me, scared):
        """
        Border stand-off breaker. A ghost cannot score by guarding, only by
        EATING an invader (it respawns far away, losing 30+ moves). So after a
        few turns of stand-off we step back 2 cells, between the door and our
        food: the enemy sees a free path, crosses, and walks into us.
        If it does not take the bait within 10 turns, guard normally again
        and do not retry for 20 turns.
        """
        mi = agent.mi
        if not USE_BAIT or scared or agent.turn < agent.bait_cooldown:
            agent.guard_turns = 0
            agent.bait_turns = 0
            return None
        waiting = []
        for e in agent.get_opponents(game_state):
            st = game_state.get_agent_state(e)
            pos = game_state.get_agent_position(e)
            if pos is not None and not st.is_pacman and not mi.is_home(pos) and mi.d(me, pos) <= 3:
                waiting.append(pos)
        if not waiting:
            agent.guard_turns = 0
            agent.bait_turns = 0
            return None
        agent.guard_turns += 1
        if agent.guard_turns < 4:
            return None                     # a short wait is normal guarding

        agent.bait_turns += 1
        if agent.bait_turns > 10:           # not taking the bait: stop trying
            agent.bait_cooldown = agent.turn + 20
            agent.guard_turns = 0
            agent.bait_turns = 0
            return None

        g = min(waiting, key=lambda p: mi.d(me, p))
        doors = [n for n in mi.neighbors[g] if mi.is_home(n)]
        if not doors:
            return None
        door = doors[0]
        our_food = agent.get_food_you_are_defending(game_state).as_list()
        # cells 2 steps behind the door, on our side; pick the one nearest our food
        spots = [c for c, dd in mi.home_dist.get(door, {}).items() if dd == 2]
        if not spots:
            return None
        if our_food:
            spot = min(spots, key=lambda c: min(mi.d(c, f) for f in our_food))
        else:
            spot = spots[0]
        return Plan('defend', 'bait', spot, door=door)

    def patrol_point(self, agent, game_state):
        mi = agent.mi
        if not mi.home_border:
            return agent.start
        our_food = agent.get_food_you_are_defending(game_state).as_list()
        best, best_v = None, INF
        for b in mi.home_border:
            v = 0.0
            for e in agent.get_opponents(game_state):
                belief = self.tracker.belief.get(e, {})
                # expected distance from this border cell to the enemy
                v += sum(p * mi.d(b, c) for c, p in belief.items()) if belief else 0
            if our_food:
                v += 0.3 * min(mi.d(b, f) for f in our_food)
            if v < best_v:
                best, best_v = b, v
        return best


#############################################################################
# SECTION 3 - JUDGE: scores states + reviews moves (corrections)            #
#############################################################################

class Judge:
    """
    Scores a (simplified) board state with ONE number, from our attacker's
    point of view. Used at the leaves of the alpha-beta search.

    Big picture of the weights:
        +120 per pellet banked at home      (terminal: reached home)
        +300 for grabbing a power capsule   (terminal: ghosts become harmless)
        -5000 for being eaten               (terminal: we lose what we carry)
        +15 per pellet carried
        -2/step to the next pellet, or -4/step to home when returning
        -400 ghost next to us, -80 ghost two steps away
        -300 we are in a dead end and the ghost can reach its exit first
        -100 no border cell we can reach before the ghost (Voronoi)
    """

    EATEN = -5000
    HOME_BASE = 100
    HOME_PER_FOOD = 120
    CAPSULE = 300

    def __init__(self, mi, food, capsules, returning):
        self.mi = mi
        self.food = food                  # list of pellets we may eat
        self.capsules = set(capsules)
        self.returning = returning        # True -> goal is to go home

    def terminal_eaten(self, carried, plies):
        return self.EATEN - 50 * carried + 10 * plies   # later death = less bad

    def terminal_home(self, carried, plies):
        return self.HOME_BASE + self.HOME_PER_FOOD * carried - 2 * plies

    def terminal_capsule(self, plies):
        return self.CAPSULE - 2 * plies

    def leaf(self, me, ghosts, carried, eaten):
        mi = self.mi
        value = 15.0 * carried

        if self.returning or carried > 0 and not any(f not in eaten for f in self.food):
            value -= 4.0 * mi.dist_home[me]
        else:
            best = 9999
            for f in self.food:
                if f not in eaten:
                    df = mi.d(me, f)
                    if df < best:
                        best = df
            if best < 9999:
                value -= 2.0 * best

        if not mi.is_home(me) and ghosts:
            for g in ghosts:
                dg = mi.d(me, g)
                if dg <= 1:
                    value -= 400
                elif dg == 2:
                    value -= 80
                else:
                    value -= 30.0 / dg
                if mi.trap_depth[me] > 0:
                    ex = mi.trap_exit[me]
                    if mi.d(g, ex) <= mi.d(me, ex) + 1:
                        value -= 300
            if not mi.safe_border_cells(me, ghosts):
                value -= 100
        return value


    # ============================================ JUDGE: review + correction
    # The second job of the Judge in the team diagram: the executor proposes
    # a move, the judge checks it and, if it is unsafe, sends back a
    # correction. Repeated corrections are fed back to the evaluator, which
    # then drops that objective for a while (see CoachedAgent._decide).

    @staticmethod
    def review(agent, game_state, plan, cell):
        """Returns (cell to play, reason for the correction or None)."""
        mi = agent.mi
        me = game_state.get_agent_position(agent.index)
        if cell is None:
            return me, None
        threats = plan.info.get('threats', [])
        attacking = plan.role in ATTACK_ROLES

        # (a) Defenders never cross the line.
        if not attacking and mi.is_home(me) and not mi.is_home(cell):
            return agent._step_towards_home_only(me, plan.objective or me), 'defender tried to cross'

        # The alpha-beta search already judged its own move against the ghosts.
        if attacking and threats and not plan.info.get('searched'):
            def ghost_dist(c):
                return min(mi.d(g, c) for g in threats)

            def goal_dist(c):
                return min([mi.d(c, g) for g in plan.goals] or [0])

            # (b) Never step from home onto a cell a ghost can hit next turn.
            if mi.is_home(me) and not mi.is_home(cell) and ghost_dist(cell) <= 2:
                agent.stuck_turns += 3
                alt = [n for n in mi.neighbors[me] if mi.is_home(n)]
                if alt:
                    entry = agent.coach.pick_entry(agent, me, threats)
                    return (min(alt, key=lambda n: mi.hd(n, entry)) if entry else me), 'crossing guarded'
                return me, 'crossing guarded'

            if not mi.is_home(cell):
                # (c) On enemy ground, never step next to a ghost if a safer move exists.
                if ghost_dist(cell) <= 1:
                    safer = [n for n in mi.moves_from[me] if mi.is_home(n) or ghost_dist(n) >= 2]
                    if safer:
                        return min(safer, key=goal_dist), 'too close to a ghost'
                # (d) Don't walk deeper into a dead end the ghost can close.
                if mi.trap_depth[cell] > mi.trap_depth[me]:
                    ex = mi.trap_exit[cell]
                    if ghost_dist(ex) <= mi.trap_depth[cell] + 1:
                        safer = [n for n in mi.moves_from[me]
                                 if n != cell and (mi.is_home(n) or mi.trap_depth[n] <= mi.trap_depth[me])]
                        if safer:
                            # head OUT of the pocket first, then towards the goal
                            return min(safer, key=lambda n: (mi.trap_depth[n], goal_dist(n))), \
                                'dead end can be closed'

        # (e) An ATTACKER pacing around the same few cells (12 turns, at most
        #     3 different cells) is stuck: feed that back to the evaluator so
        #     the next plan is different. Defenders are allowed to stand still
        #     on guard, so this does not apply to them.
        hist = list(agent.recent_positions)[-12:]
        if attacking and len(hist) == 12 and len(set(hist)) <= 3:
            if plan.mode == 'eat' and plan.objective is not None:
                agent.taboo[plan.objective] = agent.turn + 20
            else:
                food = agent.get_food(game_state).as_list()
                if food:
                    nearest = min(food, key=lambda f: mi.d(me, f))
                    agent.taboo[nearest] = agent.turn + 20
            agent.capsule_ban_until = agent.turn + 20
            if mi.is_home(me):
                agent.entry_point = agent.coach.pick_entry(agent, me, threats, avoid_here=True)
                agent.stuck_turns = 0
            agent.recent_positions.clear()
            return cell, 'attacker pacing'
        return cell, None


#############################################################################
# SECTION 4 - EXECUTOR's search tools: alpha-beta and A*                    #
#############################################################################

class SearchTimeout(Exception):
    pass


class CombatSearch:
    """
    Alpha-beta over a tiny simulation: our attacker (MAX) against the 1-2
    nearest dangerous ghosts (MIN, they try to catch us). Iterative deepening:
    search 1 move ahead, then 2, then 3 ... and keep the best answer from the
    deepest search that finished before the time budget ran out.
    """

    def __init__(self, mi, judge, deadline):
        self.mi = mi
        self.judge = judge
        self.deadline = deadline
        self.nodes = 0

    def best_move(self, me, ghosts, carried, max_depth=8):
        best_cell, best_depth = None, 0
        ordering = list(self.mi.moves_from[me])
        for depth in range(1, max_depth + 1):
            try:
                value, cell = self._root(me, tuple(ghosts), carried, depth, ordering)
            except SearchTimeout:
                break
            best_cell, best_depth = cell, depth
            # try the best move first next time -> more pruning
            ordering.remove(cell)
            ordering.insert(0, cell)
            if value <= Judge.EATEN + 1000 or value >= Judge.HOME_BASE:
                # a forced loss or a guaranteed safe return: deeper won't help
                if depth >= 2:
                    break
        return best_cell, best_depth

    def _tick(self):
        self.nodes += 1
        if self.nodes & 127 == 0 and time.time() > self.deadline:
            raise SearchTimeout()

    def _after_my_move(self, me, carried, eaten, plies):
        """Applies the effect of stepping on a cell. Returns (terminal_value or None, carried, eaten)."""
        judge, mi = self.judge, self.mi
        if me in judge.capsules:
            return judge.terminal_capsule(plies), carried, eaten
        if not mi.is_home(me) and me in judge.food and me not in eaten:
            carried += 1
            eaten = eaten | frozenset([me])
        if mi.is_home(me) and carried > 0:
            return judge.terminal_home(carried, plies), carried, eaten
        return None, carried, eaten

    def _root(self, me, ghosts, carried, depth, ordering):
        best_v, best_c = -INF, ordering[0]
        alpha, beta = -INF, INF
        for nxt in ordering:
            v = self._value_after_me(nxt, ghosts, carried, frozenset(), depth, 1, alpha, beta)
            if v > best_v:
                best_v, best_c = v, nxt
            alpha = max(alpha, v)
        return best_v, best_c

    def _value_after_me(self, me, ghosts, carried, eaten, depth, plies, alpha, beta):
        mi = self.mi
        if not mi.is_home(me) and me in ghosts:
            return self.judge.terminal_eaten(carried, plies)
        term, carried, eaten = self._after_my_move(me, carried, eaten, plies)
        if term is not None:
            return term
        return self._min(me, ghosts, carried, eaten, depth, plies, 0, alpha, beta)

    def _max(self, me, ghosts, carried, eaten, depth, plies, alpha, beta):
        self._tick()
        if depth == 0:
            return self.judge.leaf(me, ghosts, carried, eaten)
        v = -INF
        for nxt in self.mi.moves_from[me]:
            v = max(v, self._value_after_me(nxt, ghosts, carried, eaten, depth, plies + 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def _min(self, me, ghosts, carried, eaten, depth, plies, gi, alpha, beta):
        self._tick()
        mi = self.mi
        v = INF
        g = ghosts[gi]
        for nxt in mi.moves_from[g]:
            if nxt == me and not mi.is_home(me):
                val = self.judge.terminal_eaten(carried, plies)
            else:
                if mi.is_home(nxt) and not mi.is_home(g):
                    # a ghost crossing into our half would turn into a Pacman;
                    # guards don't do that while hunting us, so skip it
                    continue
                new_g = ghosts[:gi] + (nxt,) + ghosts[gi + 1:]
                if gi + 1 < len(ghosts):
                    val = self._min(me, new_g, carried, eaten, depth, plies, gi + 1, alpha, beta)
                else:
                    val = self._max(me, new_g, carried, eaten, depth - 1, plies, alpha, beta)
            if val < v:
                v = val
            if v <= alpha:
                return v
            beta = min(beta, v)
        if v == INF:     # ghost had no legal move in our model: it stays
            if gi + 1 < len(ghosts):
                return self._min(me, ghosts, carried, eaten, depth, plies, gi + 1, alpha, beta)
            return self._max(me, ghosts, carried, eaten, depth - 1, plies, alpha, beta)
        return v


def astar_first_step(mi, start, goals, danger):
    """
    A* from start to the nearest of several goal cells. Step cost is
    1 + danger[cell], so the path bends around ghosts. The heuristic
    (walking distance to the closest goal) never overestimates, so the path
    is optimal for this cost. Returns the first cell to step on.
    """
    goals = [g for g in goals if g in mi.dist]
    if not goals or start in goals:
        return None
    goal_set = set(goals)

    def h(c):
        return min(mi.d(c, g) for g in goals)

    open_heap = [(h(start), 0, start)]
    g_cost = {start: 0}
    came_from = {start: None}
    expanded = 0
    while open_heap:
        _, gc, c = heapq.heappop(open_heap)
        if gc > g_cost.get(c, INF):
            continue
        if c in goal_set:
            while came_from[c] is not None and came_from[c] != start:
                c = came_from[c]
            return c
        expanded += 1
        if expanded > 3000:
            break
        for n in mi.neighbors[c]:
            ng = gc + 1 + danger.get(n, 0)
            if ng < g_cost.get(n, INF):
                g_cost[n] = ng
                came_from[n] = c
                heapq.heappush(open_heap, (ng + h(n), ng, n))
    return None


#############################################################################
# THE AGENT - glues the five sections together                             #
#############################################################################

class CoachedAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.mi = None
        self.coach = None
        self.start = None
        self.recent_positions = deque(maxlen=12)   # SECTION 5: anti-pacing
        self.taboo = {}                            # cell -> turn until it is banned
        self.turn = 0
        self.stuck_turns = 0                       # attacker turns spent unable to cross
        self.entry_point = None                    # alternative crossing point
        self.corrections = 0                       # judge corrections in a row
        self.capsule_ban_until = 0                 # no capsule-chasing until this turn
        self.guard_turns = 0                       # turns facing a waiting enemy ghost
        self.bait_turns = 0                        # turns spent baiting
        self.bait_cooldown = 0                     # no baiting until this turn
        self.last_plan = None                      # the evaluator's latest plan

    # ------------------------------------------------------------------ setup
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.mi = MapIntel.get(game_state, self.red)
        self.coach = TeamCoach.for_team(self.red)
        # first agent of the team to register resets the shared coach
        if self.index == min(self.get_team(game_state)):
            self.coach.reset()
        if self.coach.tracker is None:
            self.coach.tracker = EnemyTracker(self.mi, game_state, self.get_opponents(game_state))

    # ------------------------------------------------------- SECTION 5 wrapper
    def choose_action(self, game_state):
        """Never crash, never time out: any error falls back to a safe move."""
        t0 = time.time()
        legal = game_state.get_legal_actions(self.index)
        try:
            action = self._decide(game_state, t0)
            if action not in legal:
                action = self._fallback(game_state, legal)
        except Exception:
            if DEBUG:
                raise
            action = self._fallback(game_state, legal)
        if DEBUG:
            dt = time.time() - t0
            CoachedAgent.max_move_time = max(getattr(CoachedAgent, 'max_move_time', 0), dt)
            if dt > 0.5:
                print('[debug] agent %d slow move: %.3fs' % (self.index, dt))
        self.recent_positions.append(game_state.get_agent_position(self.index))
        return action

    def final(self, game_state):
        if DEBUG:
            print('[debug] agent %d max move time %.3fs' % (self.index, getattr(CoachedAgent, 'max_move_time', 0)))
        CaptureAgent.final(self, game_state)

    def _fallback(self, game_state, legal):
        non_stop = [a for a in legal if a != Directions.STOP]
        return random.choice(non_stop) if non_stop else Directions.STOP

    # ------------------------------------------------------------ main brain
    def _decide(self, game_state, t0):
        """
        One turn, following the team diagram:
          tracker -> EVALUATOR (role + objective) -> EXECUTOR (move)
                  -> JUDGE (review, correction) -> action
        """
        self.turn += 1
        coach = self.coach
        tracker = coach.tracker

        # 0. Update beliefs about the enemies (SECTION 1b)
        if self.turn > 1 or self.index != 0:
            tracker.elapse((self.index - 1) % game_state.get_num_agents())
        tracker.observe(self, game_state)

        # 1. EVALUATOR: role for every agent, then this agent's objective
        coach.assign_roles(self, game_state)
        plan = coach.make_plan(self, game_state)
        self.last_plan = plan

        # 2. EXECUTOR: offensive or defensive module proposes a move
        if plan.role in ATTACK_ROLES:
            cell = self._execute_attack(game_state, plan, t0)
        else:
            cell = self._execute_defend(game_state, plan)

        # 3. JUDGE: review the move, correct it if needed
        cell, correction = Judge.review(self, game_state, plan, cell)

        # 4. Feedback: the judge keeps correcting moves towards the same
        #    pellet -> the evaluator drops that objective for a while.
        if correction:
            self.corrections += 1
            if self.corrections >= 3 and plan.mode == 'eat' and plan.objective is not None:
                self.taboo[plan.objective] = self.turn + 15
                self.corrections = 0
        else:
            self.corrections = 0

        return self._cell_to_action(game_state, cell)

    # ========================================================= EXECUTOR: offence
    def _execute_attack(self, game_state, plan, t0):
        mi = self.mi
        me = game_state.get_agent_position(self.index)
        info = plan.info
        # Close combat: alpha-beta against the nearest ghosts (SECTION 4).
        if info['fight']:
            judge = Judge(mi, info['food'], info['capsules'], info['returning'])
            search = CombatSearch(mi, judge, t0 + SEARCH_TIME_BUDGET)
            cell, depth = search.best_move(me, info['threats'][:2], info['carried'])
            if cell is not None:
                info['searched'] = True
                return cell
        # Travel: A* around ghosts to the evaluator's goal cells.
        return astar_first_step(mi, me, plan.goals, self._danger_map(info['threats']))

    # ========================================================= EXECUTOR: defence
    def _execute_defend(self, game_state, plan):
        mi = self.mi
        me = game_state.get_agent_position(self.index)
        target = plan.objective

        if plan.mode == 'hold':
            return me
        if plan.mode == 'shadow':
            # We are scared: stay 2 steps from the invader until the timer ends.
            best, best_v = me, INF
            for n in mi.moves_from[me]:
                if not mi.is_home(n):
                    continue
                dn = mi.hd(n, target)
                v = abs(dn - 2) + (100 if dn <= 1 else 0)
                if v < best_v:
                    best, best_v = n, v
            return best
        if plan.mode == 'chase':
            # Minimise distance to the invader, tie-break by cutting off its way home.
            home_escape = plan.info.get('home_escape', target)
            best, best_v = me, (INF, INF)
            for n in mi.moves_from[me]:
                if not mi.is_home(n):
                    continue
                v = (mi.hd(n, target), mi.hd(n, home_escape))
                if v < best_v:
                    best, best_v = n, v
            return best
        return self._step_towards_home_only(me, target)

    def _danger_map(self, threats):
        """Extra step cost for enemy-side cells near dangerous ghosts (used by A*)."""
        mi = self.mi
        danger = {}
        for g in threats:
            for c, dg in mi.dist[g].items():
                if dg > 3 or mi.is_home(c):
                    continue
                extra = {0: 1000, 1: 500, 2: 60, 3: 10}[dg]
                danger[c] = danger.get(c, 0) + extra
        return danger

    def _step_towards_home_only(self, me, target):
        """Greedy step towards target without crossing into the enemy half."""
        mi = self.mi
        if not mi.is_home(me):
            # We are stranded on the enemy side (e.g. just switched from raid
            # to defend): first get back home, avoiding ghosts.
            threats = [p for p in self._visible_enemy_ghosts()]
            step = astar_first_step(mi, me, mi.home_border, self._danger_map(threats))
            return step if step is not None else me
        best, best_v = me, mi.hd(me, target)
        for n in mi.neighbors[me]:
            if not mi.is_home(n):
                continue
            dn = mi.hd(n, target)
            if dn < best_v:
                best, best_v = n, dn
        return best

    def _visible_enemy_ghosts(self):
        gs = self.get_current_observation()
        out = []
        for e in self.get_opponents(gs):
            st = gs.get_agent_state(e)
            pos = gs.get_agent_position(e)
            if pos is not None and not st.is_pacman and st.scared_timer <= 2:
                out.append(pos)
        return out

    # ===================================================== SECTION 5 helpers
    def _cell_to_action(self, game_state, target_cell):
        """Turn 'go to this neighbouring cell' into N/S/E/W, with anti-pacing."""
        me = game_state.get_agent_position(self.index)
        legal = game_state.get_legal_actions(self.index)

        if target_cell is None or target_cell == me:
            action = Directions.STOP
        else:
            action = None
            for a, (dx, dy) in MOVES.items():
                if (me[0] + dx, me[1] + dy) == target_cell:
                    action = a
                    break
        if action is None or action not in legal:
            return self._fallback(game_state, legal)

        return action
