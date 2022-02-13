
package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionType;
import edu.cwru.sepia.action.LocatedAction;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.state.ResourceNode;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.util.Direction;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.*;
import java.util.function.Predicate;

public class AstarAgent extends Agent {

    /**
     * Class used to store Map information
     * Updated MapLocation class to include costs, heuristics, fvalue for A* search
     */
    class MapLocation
    {
        public int x, y;
        public float cost, heuristic;
        public float fValue; // always equal to cost + heuristic
        public MapLocation parent;

        /**
         * Original constructor (used for creating start, goal, enemy, tree locations)
         * @param x the x location of object
         * @param y the y locaiton of object
         *
         */
        public MapLocation(int x, int y, MapLocation cameFrom, float cost)
        {
            this.x = x;
            this.y = y;
        }

        /**
         * Overloaded constructor (used for nodes in the path)
         * Needs to include goal location for heuristic calculation
         * @param x The x location of the object
         * @param y The y locaiton of the object
         * @param cameFrom The parent
         * @param cost The cost to get to the location
         * @param goal Intended location used to calculate heuristic
         *
         */
        public MapLocation(int x, int y, MapLocation cameFrom, float cost, MapLocation goal)
        {
            this.x = x;
            this.y = y;

            this.parent = cameFrom;

            // Sets cost from parameter and calculates heuristic based on goal location
            this.cost = cost;
            this.heuristic = findHeuristic(goal);

            this.fValue = cost + heuristic;
        }

        /**
         * This method calculates the heuristic function using Chebsyhev Distance
         * @param goal The intended location
         * @return d which is the Chebsyhev distance
         */
        private float findHeuristic(MapLocation goal)
        {
            // Get goal x and y position from goal MapLocation argument
            int x2 = goal.x;
            int y2 = goal.y;
            
            // Calculate Chebyshev distance from current location to goal
            int d = Math.max(Math.abs(x2 - x), Math.abs(y2 - y));

            // Returns Chebyshev distance value as heuristic
            return d;

            // No need to call this method more than once -- only called when created
        }

        /**
         * This method updates the path and calculates f-values
         * @param newParent
         * @param newCost
         */
        public void updatePath(MapLocation newParent, float newCost)
        {
            parent = newParent;
            cost = newCost;

            // Recalculates total f value
            fValue = heuristic + cost;
        }
    }

    Stack<MapLocation> path;
    int footmanID, townhallID, enemyFootmanID;
    MapLocation nextLoc;

    private long totalPlanTime = 0; // nsecs
    private long totalExecutionTime = 0; //nsecs

    /**
     * Constructor to create AstarAgent
     * @param playernum Number assigned to agent
     */
    public AstarAgent(int playernum)
    {
        super(playernum);

        System.out.println("Constructed AstarAgent");
    }

    /**
     * Takes care of steps done initially
     * @param newstate
     * @param statehistory
     * @return
     */
    @Override
    public Map<Integer, Action> initialStep(State.StateView newstate, History.HistoryView statehistory) {
        // get the footman location
        List<Integer> unitIDs = newstate.getUnitIds(playernum);

        if(unitIDs.size() == 0)
        {
            System.err.println("No units found!");
            return null;
        }

        footmanID = unitIDs.get(0);

        // double check that this is a footman
        if(!newstate.getUnit(footmanID).getTemplateView().getName().equals("Footman"))
        {
            System.err.println("Footman unit not found");
            return null;
        }

        // find the enemy playernum
        Integer[] playerNums = newstate.getPlayerNumbers();
        int enemyPlayerNum = -1;
        for(Integer playerNum : playerNums)
        {
            if(playerNum != playernum) {
                enemyPlayerNum = playerNum;
                break;
            }
        }

        if(enemyPlayerNum == -1)
        {
            System.err.println("Failed to get enemy playernumber");
            return null;
        }

        // find the townhall ID
        List<Integer> enemyUnitIDs = newstate.getUnitIds(enemyPlayerNum);

        if(enemyUnitIDs.size() == 0)
        {
            System.err.println("Failed to find enemy units");
            return null;
        }

        townhallID = -1;
        enemyFootmanID = -1;
        for(Integer unitID : enemyUnitIDs)
        {
            Unit.UnitView tempUnit = newstate.getUnit(unitID);
            String unitType = tempUnit.getTemplateView().getName().toLowerCase();
            //assigns id numbers
            if(unitType.equals("townhall"))
            {
                townhallID = unitID;
            }
            else if(unitType.equals("footman"))
            {
                enemyFootmanID = unitID;
            }
            else
            {
                System.err.println("Unknown unit type");
            }
        }

        if(townhallID == -1) {
            System.err.println("Error: Couldn't find townhall");
            return null;
        }

        long startTime = System.nanoTime();
        path = findPath(newstate);
        totalPlanTime += System.nanoTime() - startTime;

        return middleStep(newstate, statehistory);
    }

    /**
     * Handles movement as agent is moving through map. Identifies end scenarios.
     * @param newstate
     * @param statehistory
     * @return
     */
    @Override
    public Map<Integer, Action> middleStep(State.StateView newstate, History.HistoryView statehistory) {
        long startTime = System.nanoTime();
        long planTime = 0;

        Map<Integer, Action> actions = new HashMap<Integer, Action>();

        if(shouldReplanPath(newstate, statehistory, path)) {
            long planStartTime = System.nanoTime();
            path = findPath(newstate);
            planTime = System.nanoTime() - planStartTime;
            totalPlanTime += planTime;
        }

        Unit.UnitView footmanUnit = newstate.getUnit(footmanID);

        int footmanX = footmanUnit.getXPosition();
        int footmanY = footmanUnit.getYPosition();

        if(!path.empty() && (nextLoc == null || (footmanX == nextLoc.x && footmanY == nextLoc.y))) {

            // start moving to the next step in the path
            nextLoc = path.pop();

            System.out.println("Moving to (" + nextLoc.x + ", " + nextLoc.y + ")");
        }

        if(nextLoc != null && (footmanX != nextLoc.x || footmanY != nextLoc.y))
        {
            int xDiff = nextLoc.x - footmanX;
            int yDiff = nextLoc.y - footmanY;

            // figure out the direction the footman needs to move in
            Direction nextDirection = getNextDirection(xDiff, yDiff);

            actions.put(footmanID, Action.createPrimitiveMove(footmanID, nextDirection));
        } else {
            Unit.UnitView townhallUnit = newstate.getUnit(townhallID);

            // if townhall was destroyed on the last turn
            if(townhallUnit == null) {
                terminalStep(newstate, statehistory);
                System.exit(0);
                return actions;

            }

            if(Math.abs(footmanX - townhallUnit.getXPosition()) > 1 ||
                    Math.abs(footmanY - townhallUnit.getYPosition()) > 1)
            {
                System.err.println("Invalid plan. Cannot attack townhall");
                totalExecutionTime += System.nanoTime() - startTime - planTime;
                return actions;
            }
            else {
                System.out.println("Attacking TownHall");
                // if no more movements in the planned path then attack
                actions.put(footmanID, Action.createPrimitiveAttack(footmanID, townhallID));
            }
        }

        totalExecutionTime += System.nanoTime() - startTime - planTime;
        return actions;
    }

    /**
     * Ouputs informaiton about timing after termination
     * @param newstate
     * @param statehistory
     */
    @Override
    public void terminalStep(State.StateView newstate, History.HistoryView statehistory) {
        System.out.println("Total turns: " + newstate.getTurnNumber());
        System.out.println("Total planning time: " + totalPlanTime/1e9);
        System.out.println("Total execution time: " + totalExecutionTime/1e9);
        System.out.println("Total time: " + (totalExecutionTime + totalPlanTime)/1e9);
    }

    @Override
    public void savePlayerData(OutputStream os) {

    }

    @Override
    public void loadPlayerData(InputStream is) {

    }

    /**
     * You will implement this method.
     *
     * This method should return true when the path needs to be replanned
     * and false otherwise. This will be necessary on the dynamic map where the
     * footman will move to block your unit.
     * 
     * You can check the position of the enemy footman with the following code:
     * state.getUnit(enemyFootmanID).getXPosition() or .getYPosition().
     * 
     * There are more examples of getting the positions of objects in SEPIA in the findPath method.
     *
     * @param state
     * @param history
     * @param currentPath
     * @return true if the next move will collide with the enemy's current or future position, false otherwise
     */
    private boolean shouldReplanPath(State.StateView state, History.HistoryView history, Stack<MapLocation> currentPath)
    {
        // If the stack is empty or the enemy doesn't exist, no need to replan path
        if (currentPath.empty() || enemyFootmanID == -1) {
            return false;
        }
    
        int enemyX, enemyY;

        // Peeks at the command issued by the enemy last turn
        int turn = state.getTurnNumber();
        Action enemyAction = history.getCommandsIssued(1, turn - 1).get(enemyFootmanID);

        // If the enemy is on the move, use its future position
        if (enemyAction != null && enemyAction.getType() == ActionType.COMPOUNDMOVE) {
            LocatedAction enemyMove = (LocatedAction) enemyAction;
            enemyX = enemyMove.getX();
            enemyY = enemyMove.getY();
        } else { // Else use its current position
            enemyX = state.getUnit(enemyFootmanID).getXPosition();
            enemyY = state.getUnit(enemyFootmanID).getYPosition();
        }

        // Lambda function to check if two MapLocations are in the same place
        Predicate<MapLocation> collides = location -> (location.x == enemyX && location.y == enemyY);
        
        // Checks if the next move on the stack will collide with the enemy
        return collides.test(currentPath.peek());
    }

    /**
     * This method is implemented for you. You should look at it to see examples of
     * how to find units and resources in Sepia.
     *
     * @param state
     * @return
     */
    private Stack<MapLocation> findPath(State.StateView state)
    {
        Unit.UnitView townhallUnit = state.getUnit(townhallID);
        Unit.UnitView footmanUnit = state.getUnit(footmanID);

        MapLocation startLoc = new MapLocation(footmanUnit.getXPosition(), footmanUnit.getYPosition(), null, 0);

        MapLocation goalLoc = new MapLocation(townhallUnit.getXPosition(), townhallUnit.getYPosition(), null, 0);

        MapLocation footmanLoc = null;
        if(enemyFootmanID != -1) {
            Unit.UnitView enemyFootmanUnit = state.getUnit(enemyFootmanID);
            footmanLoc = new MapLocation(enemyFootmanUnit.getXPosition(), enemyFootmanUnit.getYPosition(), null, 0);
        }

        // get resource locations
        List<Integer> resourceIDs = state.getAllResourceIds();
        Set<MapLocation> resourceLocations = new HashSet<MapLocation>();
        for(Integer resourceID : resourceIDs)
        {
            ResourceNode.ResourceView resource = state.getResourceNode(resourceID);

            resourceLocations.add(new MapLocation(resource.getXPosition(), resource.getYPosition(), null, 0));
        }

        return AstarSearch(startLoc, goalLoc, state.getXExtent(), state.getYExtent(), footmanLoc, resourceLocations);
    }

    /**
     * This is the method you will implement for the assignment. Your implementation
     * will use the A* algorithm to compute the optimum path from the start position to
     * a position adjacent to the goal position.
     *
     * Therefore your you need to find some possible adjacent steps which are in range 
     * and are not trees or the enemy footman.
     * Hint: Set<MapLocation> resourceLocations contains the locations of trees
     *
     * You will return a Stack of positions with the top of the stack being the first space to move to
     * and the bottom of the stack being the last space to move to. If there is no path to the townhall
     * then return null from the method and the agent will print a message and do nothing.
     * The code to execute the plan is provided for you in the middleStep method.
     *
     * As an example consider the following simple map
     *
     * F - - - -
     * x x x - x
     * H - - - -
     *
     * F is the footman
     * H is the townhall
     * x's are occupied spaces
     *
     * xExtent would be 5 for this map with valid X coordinates in the range of [0, 4]
     * x=0 is the left most column and x=4 is the right most column
     *
     * yExtent would be 3 for this map with valid Y coordinates in the range of [0, 2]
     * y=0 is the top most row and y=2 is the bottom most row
     *
     * resourceLocations would be {(0,1), (1,1), (2,1), (4,1)}
     *
     * The path would be
     *
     * (1,0)
     * (2,0)
     * (3,1)
     * (2,2)
     * (1,2)
     *
     * Notice how the initial footman position and the townhall position are not included in the path stack
     *
     * @param start Starting position of the footman
     * @param goal MapLocation of the townhall
     * @param xExtent Width of the map
     * @param yExtent Height of the map
     * @param resourceLocations Set of positions occupied by resources
     * @return Stack of positions with top of stack being first move in plan
     */
    private Stack<MapLocation> AstarSearch(MapLocation start, MapLocation goal, int xExtent, int yExtent, MapLocation enemyFootmanLoc, Set<MapLocation> resourceLocations)
    {
        // Initializing open list for A* search algorithm
    	List<MapLocation> openList = new ArrayList<MapLocation>();

    	// Maps of locations to increase efficiency of algorithm
    	boolean[][] resourceLocationMap = getLocationMap(xExtent, yExtent, resourceLocations); 
    	boolean[][] closedListLocationMap = new boolean[xExtent][yExtent]; // the closed list only needs to exist as a location map
    	
        // Start is the first node to expand, so calculate heuristic then add it to the open list
        MapLocation firstNode = new MapLocation(start.x, start.y, null, 0, goal);
    	openList.add(firstNode);

        // S denotes current node to be expanded
    	MapLocation S = null;
    	
        // A* search loop, runs until the open list is empty or we find the goal
    	while (!openList.isEmpty())
    	{
            // Chooses the the next node to be expanded (min f-value) using helper function
    		S = nextBestLocation(openList);

    		if (S.heuristic == 0) { // S.heuristic = 0 if and only if S is the goal
    			break;
    		}

            // Generates offset pairs for the 8 directions of movement
    		int[] xOffsets = {1,1,1,0,-1,-1,-1,0};
    		int[] yOffsets = {-1,0,1,1,1,0,-1,-1};
    			
            // Expands S by adding/updating neighbors to the open list
    		neighbors:
    		for (int i = 0; i < 8; i++) {
                // Looks at the location in a certain direction and performs various checks on it
                // The checks are in a specific order to optimize runtime (more likely events are checked first)
    			int x = S.x + xOffsets[i];
    			int y = S.y + yOffsets[i];
    			
    			// Check if in bounds
    	        if (x < 0 || x >= xExtent) continue;
    	        if (y < 0 || y >= yExtent) continue;
    	            	        
    	        // Check if in closed list
    			if (closedListLocationMap[x][y]) continue;

                // Checks if enemy footman or resource is already occupying location
    			if (isOccupied(x, y, enemyFootmanLoc, resourceLocationMap)) continue;

                // Sets cost of neighbor (parent cost + distance to neighbor)
    			float cost = S.cost + (float) Math.sqrt(xOffsets[i]*xOffsets[i]+yOffsets[i]*yOffsets[i]);

                // Check if in open list
    			for (MapLocation openListLocation : openList) {
        			if (openListLocation.x == x && openListLocation.y == y) {
                        // If we found a better path to this node, update it
        				if (cost < openListLocation.cost) {
        					openListLocation.updatePath(S, cost);
        				}
        			    continue neighbors;
        			}
        		}

                // All checks have passed at this point, adds the neighbor to the open list
                MapLocation newNode = new MapLocation(x, y, S, cost, goal);
                openList.add(newNode);
    		}

            // Removes S from the open list and adds it to the closed list
    		openList.remove(S);
    		closedListLocationMap[S.x][S.y] = true;
    	}

        // If S is not the goal node, we have exhausted the map without finding an eligible path
        if (S.heuristic != 0) {
            System.err.println("No available path.");
            System.exit(0);
        }
        // At this point, S is the goal node and we have found the optimal path to it

        // Creates a stack path to add the optimal path
    	Stack<MapLocation> path = new Stack<MapLocation>();
        
        // Goal should not be in the stack, so go to its parent
        S = S.parent;

        // Loop to fill the stack from goal -> start, adding each member's parent
    	while (S.x != start.x || S.y != start.y) {
    		path.push(S);
    		S = S.parent;
    	}

        // Returns the path (with nodes in correct order)
        return path;
    }
    
    /**
     * Helper method to determine the next best location (in terms of f-value)
     * in the given open list
     * @param openList
     * @return 
     */
    private MapLocation nextBestLocation(List<MapLocation> openList) {
        // Open list is never empty when this method is called, so get its first element
    	MapLocation bestLocation = openList.get(0);

        // Iterates through add locations in open list
    	for (MapLocation possibleLocation : openList) {
    		if (possibleLocation.fValue < bestLocation.fValue) { // if new minimum is found
                // Updates bestLocation to the newfound location
    			bestLocation = possibleLocation;
    		}
    	}
        
    	return bestLocation;
    }
       
    /**
     * Helper method to build a boolean map of locations,
     * to allow for fast checks in functions like isOccupied()
     * @param xExtent
     * @param yExtent
     * @param resourceLocations
     * @return a 2D boolean array that holds true in the locations of the resources
     */
    private boolean[][] getLocationMap(int xExtent, int yExtent, Set<MapLocation> resourceLocations) {
    	boolean[][] locationMap = new boolean[xExtent][yExtent];
    	
        // Iterates through all resource locations and sets the corresponding array entry to true
    	for (MapLocation location : resourceLocations) {
    		locationMap[location.x][location.y] = true; 
    	}

    	return locationMap;
    }
    
    /**
     * Helper method to determine if a given (x, y) location is occupied
     * @param x x-coordinate of space to be checked
     * @param y y-coordinate of space to be checked
     * @param enemyFootmanLoc
     * @param resourceLocationMap
     * @return true if the space contains a footman/resource
     */
    private boolean isOccupied(int x, int y, MapLocation enemyFootmanLoc, boolean[][] resourceLocationMap) {
        // Checking if location coincides with enemy footman
        if (enemyFootmanLoc != null && (x == enemyFootmanLoc.x && y == enemyFootmanLoc.y)) return true;

        // Checking if location coincides with any resources
        if (resourceLocationMap[x][y]) return true;
    	
        // At this point the space must be empty
    	return false;
    }

    /**
     * Primitive actions take a direction (e.g. Direction.NORTH, Direction.NORTHEAST, etc)
     * This converts the difference between the current position and the
     * desired position to a direction.
     *
     * @param xDiff Integer equal to 1, 0 or -1
     * @param yDiff Integer equal to 1, 0 or -1
     * @return A Direction instance (e.g. SOUTHWEST) or null in the case of error
     */
    private Direction getNextDirection(int xDiff, int yDiff) {

        // figure out the direction the footman needs to move in
        if(xDiff == 1 && yDiff == 1)
        {
            return Direction.SOUTHEAST;
        }
        else if(xDiff == 1 && yDiff == 0)
        {
            return Direction.EAST;
        }
        else if(xDiff == 1 && yDiff == -1)
        {
            return Direction.NORTHEAST;
        }
        else if(xDiff == 0 && yDiff == 1)
        {
            return Direction.SOUTH;
        }
        else if(xDiff == 0 && yDiff == -1)
        {
            return Direction.NORTH;
        }
        else if(xDiff == -1 && yDiff == 1)
        {
            return Direction.SOUTHWEST;
        }
        else if(xDiff == -1 && yDiff == 0)
        {
            return Direction.WEST;
        }
        else if(xDiff == -1 && yDiff == -1)
        {
            return Direction.NORTHWEST;
        }

        System.err.println("Invalid path. Could not determine direction");
        return null;
    }
}
