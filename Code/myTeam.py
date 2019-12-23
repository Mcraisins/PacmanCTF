# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import itertools
import operator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent', numTraining=0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

flowMap = {}
careabout = {}
paintRed = []
avoid = {}
exits = {}
Paintblue = []
paintGreen = []
paintPurple = []

def setMaxFlow(pos1, pos2, flow, positions=None):
  if pos2 == (-1,-1): return
  if (pos1, pos2) not in flowMap:
    flowMap[(pos1, pos2)] = flow
    flowMap[(pos2, pos1)] = flow
    
    npos1 = (width - pos1[0] - 1, height - pos1[1] - 1)
    npos2 = (width - pos2[0] - 1, height - pos2[1] - 1)
    flowMap[(npos1, npos2)] = flow
    flowMap[(npos2, npos1)] = flow
      

def deadEndVertex(pos, positions):
  from collections import deque
  queue = deque([pos])
  deadends = []
  retval = 0

  ex = ()
    
  while len(queue) != 0:
    a = queue.popleft()
    if a == (-1, -1): continue
    if a[0] == (width/2-1): ex = a; break

    neighbors = [tuple(map(operator.add, a, off)) for off in [(-1,0),(1,0),(0,-1),(0,1)] if tuple(map(operator.add, a, off)) in positions]
    if len(neighbors) > 1: ex = a; break

    if neighbors == []: return
    
    neighbor = neighbors[0]
    positions[positions.index(a)] = (-1,-1)
    paintRed.append(a)
    deadends.append(a)
    for pos2 in positions:
      setMaxFlow(a, pos2, 1, positions)
      if neighbor not in queue:
        queue.append(neighbor)

  for d in deadends:
    avoid[d] = ex
    nd = (width - d[0] - 1, height - d[1] - 1)
    nex = (width - ex[0] - 1, height - ex[1] - 1)
    avoid[nd] = nex
    

def createFlowCache(gameState):
  flowMap.clear()
  walls = gameState.getWalls()
  wallList = walls.asList()
  global width, height
  width = walls.width
  height = walls.height
  positions = [pos for pos in itertools.product(range(walls.width/2), range(walls.height)) if pos not in wallList]
  flowGraph = Graph(positions)

  for pos1 in positions:
    for off in [(-1,0), (1,0), (0,-1), (0,1)]:
      newpos = tuple(map(operator.add, pos1, off))
      if newpos not in wallList and newpos[0] < walls.width/2:
        flowGraph.addEdge(pos1, newpos, 1)
        import copy

  for pos in positions:
    deadEndVertex(pos,positions)
  positions = [pos for pos in positions if pos != (-1, -1)]
  
  for pos2 in positions:
    pos1 = careabout[pos2]
    if pos1 == pos2: continue
    if (pos1, pos2) in flowMap: continue
    tempFlow = copy.deepcopy(flowGraph)
    max = tempFlow.getMaxFlow(tempFlow.vind(pos1), tempFlow.vind(pos2))
    setMaxFlow(pos1, pos2, max, positions)

def exFind(pos, flow, positions):
  from collections import deque
  while len(pos) != 0:
    queue = deque([pos[0]])
    visited = {}
    visited[pos[0]] = True
    found = []
    ex = []
    while queue:
      u = queue.popleft()

      if maxFlow(u) == (flow+1) or u[0] == (width/2-1):
        if u not in ex:
          ex.append(u)
          continue

      if maxFlow(u) == flow:
        if u in pos:
          pos.remove(u)
          found.append(u)
      
          neighbors = [tuple(map(operator.add, u, off)) for off in [(-1,0),(1,0),(0,-1),(0,1)] if tuple(map(operator.add, u, off)) in positions]
      
          for n in neighbors:
            if n not in visited.keys():
              visited[n] = True
              queue.append(n)

    result = {}
    for f in found:
      nf = (width - f[0] - 1, height - f[1] - 1)
      if f not in result: result[f] = []
      if nf not in result: result[nf] = []
      for e in ex:
        ne = (width - e[0] - 1, height - e[1] -1)
        result[f].append(e)
        result[nf].append(ne)
        exits.update(result)
              
def maxFlow(pos1, red=True):
  return flowMap[(careabout[pos1], pos1)]

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  
  def paintMap(self):
    drawn = []
    positions = flowMap.keys()

    for pos2 in itertools.product(range(width), range(height)):
      if pos2 not in careabout: continue
      pos1 = careabout[pos2]
      if pos1 == pos2: continue
      self.debugDraw([pos2],[[0,0,0.1],[0,0,0.5],[0,0.5,0.5],[0.5,0.5,0]][flowMap[(pos1,pos2)] - 1] , clear=False)
#    self.debugDraw(avoid.keys(), [1,0,0], clear=True)
#    self.debugDraw(exits.keys(), [1,1,0], clear=False)
#    for entry in avoid:
#      self.debugDraw(avoid[entry], [1,1,0], clear=False)
#    self.debugDraw(avoid.values(), [0,1,1], clear=False)
    for entry in exits:
      for ex in exits[entry]:
        self.debugDraw(ex, [0,1,0], clear=False)

#    print exits
  
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  held_dots = 0
  
  tooclose = 5
  retreat = False 
  target = ()
  def registerInitialState(self, gameState):

    careabout.clear()
    flowMap.clear()
    avoid.clear()

    exits.clear()
    self.start = gameState.getAgentPosition(self.index)
    self.prevPos = self.start
    self.prevFood = self.getFood(gameState)
    CaptureAgent.registerInitialState(self, gameState)

    walls = gameState.getWalls()
    wallList = walls.asList()
    global width, height
    width = walls.width
    height = walls.height
    positions = [pos for pos in itertools.product(range(walls.width/2), range(walls.height)) if pos not in wallList]
    for pos1 in positions:
      if pos1[0] == walls.width/2 - 1:
        careabout[pos1] = pos1
        newpos1 = (walls.width - pos1[0] - 1, walls.height - pos1[1] - 1)
        careabout[newpos1] = newpos1
      else:
        dist = sys.maxint
        for pos2 in positions:
          if pos2[0] == walls.width/2 - 1 and pos1 != pos2:
            temp = self.getMazeDistance(pos1,pos2)
            if temp < dist:
              dist = temp
              careabout[pos1] = pos2

              newpos1 = (walls.width - pos1[0] - 1, walls.height - pos1[1] - 1)
              newpos2 = (walls.width - pos2[0] - 1, walls.height - pos2[1] - 1)
              careabout[newpos1] = newpos2
    
    createFlowCache(gameState)

    result = {}
    for pos1 in positions:
      if pos1[0] == walls.width/2 - 1:
        if pos1 not in result: result[pos1] = []
        setMaxFlow(pos1, pos1, 5)
        
        newpos1 = (walls.width - pos1[0] - 1, walls.height - pos1[1] - 1)
        if newpos1 not in result: result[newpos1] = []
        setMaxFlow(newpos1, newpos1, 5)
        
        for pos2 in positions:
          if pos2[0] == walls.width/2-1 and pos2 != pos1:
            pos2 = (walls.width - pos2[0] - 1, walls.height - pos2[1] - 1)
            result[pos1].append(pos2)
            newpos2 = (walls.width - pos2[0] - 1, walls.height - pos2[1] - 1)
            result[newpos1].append(newpos2)
    exits.update(result)
    
    flowone = []
    flowtwo = []
    flowthree = []
    flowfour = []
    for pos in positions:
      if pos not in avoid:
        flow = maxFlow(pos)
        if flow == 1:
          flowone.append(pos)
        if flow == 2:
          flowtwo.append(pos)
        if flow == 3:
          flowthree.append(pos)
        if flow == 4:
          flowfour.append(pos)
    exFind(flowone, 1, positions)
    exFind(flowtwo, 2, positions)
    exFind(flowthree, 3, positions)
    exFind(flowfour, 4, positions)

    self.paintMap()
    self.setupDefense(gameState)
#    self.debugDraw(paintRed, [1,0,0], False)
    # initialize stuff for the defense agent
  def setupDefense(self, gameState):
    global eset
    eset = []
    team = 0 if gameState.isOnRedTeam(self.index) else 1
    for entry in exits:
      for e in exits[entry]:
        if team == 0 and e[0] < width / 2 and e not in eset: eset.append(def_exit(e, 0, 0))
        elif team == 1 and e[0] >= width / 2 and e not in eset: eset.append(def_exit(e,0,0))

    for entry in exits:
      for e in exits[entry]:
        for de in eset:
          if de.e == e:
            de.c += 1.0

    for de1 in eset:
      for de2 in eset:
        if de1.e in exits and de2.e in exits[de1.e]: de2.c += de1.c

    for de in eset:
      if de.e[0] in [width / 2 - 1, width / 2]:
        de.c /= 5

    total = 0.0
    for de in eset:
      total += de.c

    for de in eset:
      de.c /= total


  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    if gameState.getAgentState(self.index).getPosition()[0] == (1 if gameState.isOnRedTeam(self.index) else 32):
      self.held_dots = 0

    # do I need to add to my dots?
    myPos = gameState.getAgentState(self.index).getPosition()
    self.held_dots += 1 if self.prevFood[int(myPos[0])][int(myPos[1])] != 0 else 0
    self.prevFood = self.getFood(gameState)

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    team = 0 if successor.isOnRedTeam(self.index) else 1


    myPos = gameState.getAgentState(self.index).getPosition()
    newPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    close = [a for a in enemies if a.getPosition() != None and a.scaredTimer == 0 and
             self.getMazeDistance(newPos, a.getPosition()) <= self.tooclose]
    closest = [a for a in close if a.getPosition() != None and self.getMazeDistance(newPos, a.getPosition()) <= 1]

#    self.debugDraw([(16+team,y) for y in range(1,16)], [1,0,0] if team == 0 else [0,0,1])

    if len(close) > 0 and self.held_dots > 0:
      self.retreat = True 
    if myPos[0] == 16+team or self.held_dots == 0:
      self.retreat = False
      self.held_dots = 0

    # Compute distance to the nearest food
    if not self.retreat and len(foodList) > 0: # This should always be True,  but better safe than sorry
      mloc = ()
      minDistance = sys.maxint 
      for food in foodList:
        if self.getMazeDistance(newPos, food) < minDistance:
          mloc = food
          minDistance = self.getMazeDistance(newPos, food)
      target = mloc
      features['distanceToTarget'] = minDistance
    else:
      mloc = ()
      maxDistance = -1
      if myPos in avoid:
        mloc = avoid[myPos]
        maxDistance = self.getMazeDistance(newPos, mloc)
      else:
        if len(close) == 0:
          if myPos == self.target or self.target == ():
            maxDistance = sys.maxint
            if myPos in exits:
              for e in exits[myPos]:
                if self.getMazeDistance(e, newPos) < maxDistance:
                  mloc = e
                  maxDistance = self.getMazeDistance(e, newPos)
          else:
            mloc = self.target
            maxDistance = self.getMazeDistance(mloc, myPos)
        else:
          # check if there's an exit you're closer to
          if myPos in exits:
            for e in exits[myPos]:
              if self.getMazeDistance(e, close[0].getPosition()) - self.getMazeDistance(e, myPos):
                mloc = e
                maxDistance = self.getMazeDistance(newPos, mloc)
                break
          # otherwise go to the ghost's farthest exit
          if mloc != () and len(close) > 0:
            for e in exits[myPos]:
              if self.getMazeDistance(e, close[0].getPosition()) > maxDistance:
                mloc = e
                maxDistance = self.getMazeDistance(e, close[0].getPosition())
            maxDistance = self.getMazeDistance(mloc, newPos)

      if len(self.getCapsules(successor)) != 0:
        features['distanceToPacPellet'] = min([self.getMazeDistance(newPos, pellet) for pellet in self.getCapsules(successor)])
      else:
        features['distanceToPacPellet'] = 0
                      
      target = mloc
      features['distanceToTarget'] = maxDistance
    if len(closest) > 0:
      features['death'] = 1
    else:
      features['death'] = 0

    if self.retreat and newPos in avoid:
      features['death'] = 1

    if newPos == self.prevPos:
      features['repeat'] = 1
      
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToTarget': -2, 'death': -1000000, 'repeat': -10, 'distanceToPacPellet': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  target = ()
  tooclose = 5
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    #    self.debugDraw([(x,y) for x,y in itertools.product(range(1,33),range(1,17)) if distanceCalculator.manhattanDistance(myPos, (x,y)) <= 5],[0,0,1], clear=True)
    if len(invaders) > 0:
      # self.debugDraw([a.getPosition()], [1,0,0], clear=True)
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    
    myPos = gameState.getAgentState(self.index).getPosition()
    newPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    close = [a for a in enemies if a.getPosition() != None and a.scaredTimer == 0 and
             self.getMazeDistance(newPos, a.getPosition()) <= self.tooclose]
    closest = [a for a in close if a.getPosition() != None and self.getMazeDistance(newPos, a.getPosition()) <= 1]
    
    if features['onDefense']: 
      if self.target == () or myPos == self.target:
        self.newtarget();
        try:
          features['distanceToTarget'] = self.getMazeDistance((int(newPos[0]),int(newPos[1])), (int(self.target[0]),int(self.target[1])))
        except(e):
          features['distanceToTarget'] = 30

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distanceToTarget': -1}

  def newtarget(self):
    num = random.random()
    for dec in eset:
      if num < dec.c:
        self.target = dec.e
        return
      else:
        num -= dec.c
    self.target = eset[len(eset)-1].e
    return 
    
class Graph:
    def __init__(self,V): 
      self.vert = V
      self.graph = [[0 for v in V] for v in V]
      self.ROW = len(V) 

    def addEdge(self, u, v, w):
      self.graph[self.vind(u)][self.vind(v)] = w

    def vind(self, v):
      for i in range(len(self.vert)):
        if self.vert[i] == v: return i
      return -1

    '''Returns true if there is a path from source 's' to sink 't' in 
    residual graph. Also fills parent[] to store the path '''
    def BFS(self,s, t, parent): 
      
        # Mark all the vertices as not visited 
        visited =[False]*(self.ROW) 
        
        # Create a queue for BFS 
        queue=[] 
        
        # Mark the source node as visited and enqueue it 
        queue.append(s) 
        visited[s] = True
        
         # Standard BFS Loop 
        while queue: 
          
            #Dequeue a vertex from queue and print it 
            u = queue.pop(0) 
            
            # Get all adjacent vertices of the dequeued vertex u 
            # If a adjacent has not been visited, then mark it 
            # visited and enqueue it 
            for ind, val in enumerate(self.graph[u]): 
                if visited[ind] == False and val > 0 : 
                  queue.append(ind) 
                  visited[ind] = True
                  parent[ind] = u 
                  
        # If we reached sink in BFS starting from source, then return 
        # true, else false 
        return True if visited[t] else False    # Returns tne maximum flow from s to t in the given graph 

    def FordFulkerson(self, source, sink): 
      
        # This array is filled by BFS and to store path 
        parent = [-1]*(self.ROW) 
        
        max_flow = 0 # There is no flow initially 
        
        # Augment the flow while there is path from source to sink 
        while self.BFS(source, sink, parent) : 
          
            # Find minimum residual capacity of the edges along the 
          # path filled by BFS. Or we can say find the maximum flow 
          # through the path found. 
            path_flow = float("Inf") 
            s = sink 
            while(s !=  source): 
              path_flow = min (path_flow, self.graph[parent[s]][s]) 
              s = parent[s] 
              
            # Add path flow to overall flow 
            max_flow +=  path_flow 
            
            # update residual capacities of the edges and reverse edges 
            # along the path 
            v = sink 
            while(v !=  source): 
              u = parent[v] 
              self.graph[u][v] -= path_flow 
              self.graph[v][u] += path_flow 
              v = parent[v] 
              
        return max_flow 

    def getMaxFlow(self, s, t):
        return self.FordFulkerson(s, t)

class def_exit:
  def __init__(self, e, c, v):
    self.e = e
    self.c = c
    self.v = v

