

import numpy as np
import pandas as pd
import time
from kaggle_environments import make
from kaggle_environments.envs.halite.helpers import *

PRINT_MODE = False # print info text used for debugging/improving algo
FACTOR_NEAREST_SHIPYARD = 0.25 # if ship is further than all shipyards by this value*board_size than we build a new shipyard
ENEMY_DIST_THRESHOLD = 2 # if enemy ship is x moves away, initiate escape maneuver
HALITE_THRESHOLD = 500 # send ship to closest shipyard once this amount of halite is collected
HALITE_TO_SHIP_THRESHOLD = 1400 # ratio of halite to ship metric used to determine if we need more ships

def shipyard_halite_density(board):
    me = board.current_player
    board_size = board.configuration.size
    num_of_shipyards = len(me.shipyards)
    
    box_size = int(FACTOR_NEAREST_SHIPYARD * board_size)
    box_range = range(-box_size,box_size)
    
#     print(board_size,box_size,box_range)
    
    shipyard_list = []

    for shipyard in me.shipyards:
        shipyard_x, shipyard_y = shipyard.position
        
        ships_in_area = 1
        halite_in_area = 0
        halite_to_ship_ratio = 0
        
        # ships in area
        for ship in me.ships:
            ship_x,ship_y = ship.position
            dist = abs(shipyard_x-ship_x) + abs(shipyard_y-ship_y) # calculates manhattan distance
            if dist < box_size:
                ships_in_area += 1
                
        # halite in area
        for i in box_range:
            for j in box_range:
                halite_in_area += board[shipyard.position.translate((i,j),board_size)].halite
        
        halite_to_ship_ratio = halite_in_area/ships_in_area
        
        shipyard_list.append({'shipyard.id':shipyard.id,'shipyard.position':shipyard.position,'halite_in_area':round(halite_in_area,2),'ships_in_area':ships_in_area,'halite_to_ship_ratio':round(halite_to_ship_ratio,2)})
        
    df_shipyard_density = pd.DataFrame(shipyard_list)
    df_shipyard_density = df_shipyard_density.sort_values('halite_to_ship_ratio',ascending=False)
    
    return df_shipyard_density
            
def ship_dispatcher(board):
    df_ship_dispatch = pd.DataFrame(columns=['ship.id','ship.halite','ship.status','ship.position','target_ship_x','target_ship_y','move1','move2','move3','move4','move1_pos','move2_pos','move3_pos','move4_pos','move1_hal','move2_hal','move3_hal','move4_hal','total_hal'])
    board_size = board.configuration.size
#     ship = board.current_player.ships[0]

    me = board.current_player
    
    # moves = ['stay','up','down','left','right']
    moves = [(0,0),(0,1),(0,-1),(-1,0),(1,0)]
    
    free_ships = [ship for ship in me.ships if ship.next_action == None]
    
    ship_conversion_list = []
    # if ship needs to travel 20% of map size to get to nearest shipyard, create a new shipyard
    if (me.halite) >= 500:
        for ship in free_ships:
            nearest_shipyard_dist,nearest_shipyard_pos = nearest_shipyard(ship,ship_conversion_list,board)
            if nearest_shipyard_dist > board_size*FACTOR_NEAREST_SHIPYARD:
                # convert ship to shipyard
                ship.next_action = ShipAction.CONVERT
                ship_conversion_list.append(ship.position)
                
                if PRINT_MODE:
                    print(f'ACTIVATE SHIP CONVERSION -- ship.id: {ship.id}')
    
    free_ships = [ship for ship in me.ships if ship.next_action == None]
    shipyard_pos_list = [shipyard.position for shipyard in me.shipyards]

    for ship in free_ships:
        ship_x,ship_y = ship.position
        
        # return to nearest shipyard if halite threshold has been met
        if ship.halite > HALITE_THRESHOLD:
            nearest_shipyard_dist,nearest_shipyard_pos = nearest_shipyard(ship,ship_conversion_list,board)
            
            shipyard_x,shipyard_y = nearest_shipyard_pos
            
            if ship_x < shipyard_x:
                move1_pos = ship.position.translate((1,0),board_size)
            elif ship_x > shipyard_x:
                move1_pos = ship.position.translate((-1,0),board_size)
            elif ship_y < shipyard_y:
                move1_pos = ship.position.translate((0,1),board_size)
            elif ship_y > shipyard_y:
                move1_pos = ship.position.translate((0,-1),board_size)

            if move1_pos in df_ship_dispatch.move1_pos:
                # keep ship in same position to avoid collision with allied ship
                dispatch_dict = {'ship.id':ship.id,'ship.halite': ship.halite, 'ship.status': 'DEPOSIT', 'ship.position': ship.position,'target_ship_x':None, 'target_ship_y': None, 'move1': None,'move2': None,'move3': None,'move4': None, 'move1_pos': ship.position,'move2_pos': None,'move3_pos': None,'move4_pos': None,'move1_hal': None,'move2_hal': None,'move3_hal': None,'move4_hal': None,'total_hal': None}
                df_ship_dispatch.loc[df_ship_dispatch.shape[0]] = dispatch_dict
                ship.next_action = None
            else:                
                # if no ship conflicts with move1_pos (next position of all ships), we send current ship to intended position
                dispatch_dict = {'ship.id':ship.id,'ship.halite': ship.halite, 'ship.status': 'DEPOSIT', 'ship.position': ship.position,'target_ship_x':None, 'target_ship_y': None, 'move1': None,'move2': None,'move3': None,'move4': None, 'move1_pos': move1_pos,'move2_pos': None,'move3_pos': None,'move4_pos': None,'move1_hal': None,'move2_hal': None,'move3_hal': None,'move4_hal': None,'total_hal': None}
                df_ship_dispatch.loc[df_ship_dispatch.shape[0]] = dispatch_dict
                ship.next_action = move_ship(ship.position,move1_pos,board_size)

    free_ships = [ship for ship in me.ships if ship.next_action == None]
    
    for ship in free_ships:
        halite_list = []
        ship_x,ship_y = ship.position

#         count = 1
        t1 = time.time()
        ### first move
        for move1 in moves:
            move1_pos = ship.position.translate(move1,board_size)
            if move1_pos in shipyard_pos_list+df_ship_dispatch.move1_pos.to_list(): # avoid running into shipyards and other ships
                continue

            if move1 == (0,0):
                move1_hal = board[move1_pos].halite*0.25
            else:
                move1_hal = 0

            ### second move
            for move2 in moves:
                move2_pos = move1_pos.translate(move2,board_size)
                if move2_pos in shipyard_pos_list+df_ship_dispatch.move2_pos.to_list(): # avoid running into shipyards and other ships
                    continue

                if move2 == (0,0):
                    if move1 == (0,0):
                        move2_hal = move1_hal*0.75
                    else:
                        move2_hal = board[move2_pos].halite*0.25
                else:
                    move2_hal = 0

                ### third move
                for move3 in moves:
                    move3_pos = move2_pos.translate(move3,board_size)
                    if move3_pos in shipyard_pos_list+df_ship_dispatch.move3_pos.to_list(): # avoid running into shipyards and other ships
                        continue

                    if move3 == (0,0):
                        if move2 == (0,0):
                            move3_hal = move2_hal*0.75
                        else:
                            move3_hal = board[move3_pos].halite*0.25
                    else:
                        move3_hal = 0

                    ### fourth move
                    for move4 in moves:
                        move4_pos = move3_pos.translate(move4,board_size)
                        if move4_pos in shipyard_pos_list+df_ship_dispatch.move4_pos.to_list(): # avoid running into shipyards and other ships
                            continue

                        if move4 == (0,0):
                            if move3 == (0,0):
                                move4_hal = move3_hal*0.75
                            else:
                                move4_hal = board[move4_pos].halite*0.25
                        else:
                            move4_hal = 0

                        total_hal = move1_hal + move2_hal + move3_hal + move4_hal

    #                     action_dict = {'ship.id':ship.id,'ship.halite': ship.halite, 'ship.position': ship.position,'move1': move1,'move2': move2,'move3': move3,'move4': move4, 'move1_pos': move1_pos,'move2_pos': move2_pos,'move3_pos': move3_pos,'move4_pos': move4_pos,'move1_hal': move1_hal,'move2_hal': move2_hal,'move3_hal': move3_hal,'move4_hal': move4_hal,'total_hal': total_hal}
                        dispatch_dict = {'ship.id':ship.id,'ship.halite': ship.halite, 'ship.status': 'COLLECT', 'ship.position': ship.position, 'move1': move1,'move2': move2,'move3': move3,'move4': move4, 'move1_pos': move1_pos,'move2_pos': move2_pos,'move3_pos': move3_pos,'move4_pos': move4_pos,'move1_hal': move1_hal,'move2_hal': move2_hal,'move3_hal': move3_hal,'move4_hal': move4_hal,'total_hal': total_hal}
                        halite_list.append(dispatch_dict)

        df_halite = pd.DataFrame(halite_list)
        df_halite = df_halite.sort_values('total_hal',ascending=False)
        t2 = time.time()

        if len(df_halite) > 0:
            best_halite_idx = df_halite.index[0]
        else:
            print('len(df_halite) == 0')
        df_ship_dispatch.loc[df_ship_dispatch.shape[0]] = df_halite.loc[best_halite_idx]
        ship.next_action = move_ship(ship.position,df_halite.loc[best_halite_idx,'move1_pos'],board_size)

    return df_ship_dispatch,ship_conversion_list

# Returns best direction to move from one position (start_loc) to another (end_loc)
# Example: If I'm at pos 0 and want to get to pos 55, which direction should I choose?
def move_ship(start_loc, end_loc, board_size):
    start_x, start_y = divmod(start_loc[0],board_size), divmod(start_loc[1],board_size)
    end_x, end_y = divmod(end_loc[0],board_size), divmod(end_loc[1],board_size)
    if start_y < end_y: return ShipAction.NORTH
    if start_y > end_y: return ShipAction.SOUTH
    if start_x < end_x: return ShipAction.EAST
    if start_x > end_x: return ShipAction.WEST

# returns manhattan distance of nearest friendly shipyard
def nearest_shipyard(ship,ship_conversion_list,board):
    dist_list = []
    me = board.current_player
    shipyard_positions = [shipyard.position for shipyard in me.shipyards] + ship_conversion_list
    for shipyard_position in shipyard_positions:
        ship_x,ship_y = ship.position
        shipyard_x,shipyard_y = shipyard_position

        dist = abs(shipyard_x-ship_x) + abs(shipyard_y-ship_y) # calculates manhattan distance
        
        dist_list.append({'shipyard_pos':shipyard_position,'shipyard_dist':dist})
    
    nearest_shipyard_dist = 999999
    nearest_shipyard_pos = None
    for dist in dist_list:
        if dist['shipyard_dist'] < nearest_shipyard_dist:
            nearest_shipyard_dist = dist['shipyard_dist']
            nearest_shipyard_pos = dist['shipyard_pos']

    return nearest_shipyard_dist,nearest_shipyard_pos

# Directions a ship can move
directions = [ShipAction.NORTH, ShipAction.EAST, ShipAction.SOUTH, ShipAction.WEST]

# Returns the commands we send to our ships and shipyards
def agent(obs, config):
    board_size = config.size
    board = Board(obs, config)
    
    ship_spawn = 0
    shipyard_spawn = 0
    
    me = board.current_player
    
    board_halite_sum = round(sum(board.observation['halite']),2)
    if len(me.ships) > 0:
        halite_to_ship_ratio = round(board_halite_sum/len(me.ships),2)
    else:
        halite_to_ship_ratio = 999999

    # If there are no shipyards, convert first ship into shipyard.
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        shipyard_spawn += 1
        me.ships[0].next_action = ShipAction.CONVERT
        
    # If we have halite and episode is less than 75% complete, spawn a ship
    if (me.halite >= 500) and (len(me.shipyards) > 0) and (halite_to_ship_ratio > HALITE_TO_SHIP_THRESHOLD) and (board.step < board.configuration.episode_steps*0.75):
        ship_spawn += 1
        
        if len(me.shipyards) <= 1:
            me.shipyards[0].next_action = ShipyardAction.SPAWN
        else:
            # determine which shipyard has the highest halite to ship density and spawn ship there
            df_shipyard_density = shipyard_halite_density(board)
            best_idx = df_shipyard_density.index[0]
            best_shipyard_id = df_shipyard_density.loc[best_idx,'shipyard.id']
            best_ships_in_area = df_shipyard_density.loc[best_idx,'ships_in_area']
            best_halite_in_area = df_shipyard_density.loc[best_idx,'halite_in_area']
            best_shipyard_halite_to_ship_ratio = df_shipyard_density.loc[best_idx,'halite_to_ship_ratio']

            if PRINT_MODE:
                print(f'best_shipyard_id: {best_shipyard_id}, ships_in_area: {best_ships_in_area}, halite_in_area: {best_halite_in_area}, halite_to_ship_ratio: {best_shipyard_halite_to_ship_ratio}')
            
            for shipyard in me.shipyards:
                if shipyard.id == best_shipyard_id:
                    shipyard.next_action = ShipyardAction.SPAWN
                    break
            
            # print(df_shipyard_density)

    # HALITE COLLECTION/DEPOSIT MANAGER
    df_ship_dispatch,ship_conversion_list = ship_dispatcher(board) 
    shipyard_spawn += len(ship_conversion_list)
    
    # ENEMY DECISION MANAGER
    enemy_ships = [ship for ship in board.ships if board.ships[ship].player != board.current_player]
    if len(enemy_ships) > 0:
        for ship in me.ships:
            if ship.next_action == ShipAction.CONVERT:
                continue
            
            ship_x,ship_y = ship.position
            ship_halite = ship.halite

            # CREATE A MAP OF ENEMY SHIPS IN AREA
            enemy_ship_list = []
            for enemy_ship in enemy_ships:
                enemy_ship_x,enemy_ship_y = board.ships[enemy_ship].position
                enemy_ship_halite = board.ships[enemy_ship].halite

                dist = abs(enemy_ship_x-ship_x) + abs(enemy_ship_y-ship_y) # calculates manhattan distance

                enemy_ship_list.append({'enemy_ship_position':(enemy_ship_x,enemy_ship_y),'enemy_ship_dist':dist,'enemy_ship_halite':enemy_ship_halite})

            df_enemy_ship = pd.DataFrame(enemy_ship_list)
            df_enemy_ship = df_enemy_ship.sort_values('enemy_ship_dist',ascending=True)
            closest_enemy_idx = df_enemy_ship.index[0]
            enemy_ship_dist = df_enemy_ship.loc[closest_enemy_idx,'enemy_ship_dist']
            enemy_ship_halite = df_enemy_ship.loc[closest_enemy_idx,'enemy_ship_halite']
            enemy_ship_position = df_enemy_ship.loc[closest_enemy_idx,'enemy_ship_position']
            
            # ATTACK ENEMY SHIP
            if (enemy_ship_dist <= ENEMY_DIST_THRESHOLD*2) and (enemy_ship_halite > ship.halite*2): # if enemy ship has 2 times more halite than our current ship, we attack
                if ship.id in df_ship_dispatch['ship.id']:
                    ship_idx = df_ship_dispatch[df_ship_dispatch['ship.id'] == ship.id].index[0]
                else:
                    ship_idx = df_ship_dispatch.shape[0]
                    df_ship_dispatch.loc[ship_idx,'ship.id'] = ship.id
                    df_ship_dispatch.loc[ship_idx,'ship.halite'] = ship.halite
                    df_ship_dispatch.loc[ship_idx,'ship.position'] = ship.position

                df_ship_dispatch.loc[ship_idx,'ship.status'] = 'ATTACK'
                df_ship_dispatch.loc[ship_idx,'target_ship_x'] = enemy_ship_position[0]
                df_ship_dispatch.loc[ship_idx,'target_ship_y'] = enemy_ship_position[1]

                if ship_x < enemy_ship_x:
                    move1_pos = ship.position.translate((1,0),board_size)
                elif ship_x > enemy_ship_x:
                    move1_pos = ship.position.translate((-1,0),board_size)
                elif ship_y < enemy_ship_y:
                    move1_pos = ship.position.translate((0,1),board_size)
                elif ship_y > enemy_ship_y:
                    move1_pos = ship.position.translate((0,-1),board_size)

                if move1_pos in df_ship_dispatch.move1_pos:
                    # keep ship in same position to avoid collision with allied ship
                    df_ship_dispatch.loc[ship_idx,'move1_pos'] = ship.position
                    ship.next_action = None
                else:                
                    # if no ship conflicts with move1_pos (next position of all ships), we send current ship to intended position
                    df_ship_dispatch.loc[ship_idx,'move1_pos'] = move1_pos
                    ship.next_action = move_ship(ship.position,move1_pos,board_size)
                if PRINT_MODE:
                    print(f'ATTACK ENEMY SHIP -- enemy ship distance: {enemy_ship_dist}, enemy ship halite: {enemy_ship_halite}; allied ship halite: {ship.halite}')
                
            # ESCAPE ENEMY SHIP
            if (enemy_ship_dist <= ENEMY_DIST_THRESHOLD) and (enemy_ship_halite < ship.halite*2): # if enemy ship has 3 times more halite than our current ship, we attack
                ship_idx = df_ship_dispatch[df_ship_dispatch['ship.id'] == ship.id].index[0]
                df_ship_dispatch.loc[ship_idx,'ship.status'] = 'ESCAPE'
                df_ship_dispatch.loc[ship_idx,'target_ship_x'] = None
                df_ship_dispatch.loc[ship_idx,'target_ship_y'] = None

                if ship_x > enemy_ship_x:
                    move1_pos = ship.position.translate((1,0),board_size)
                elif ship_x < enemy_ship_x:
                    move1_pos = ship.position.translate((-1,0),board_size)
                elif ship_y > enemy_ship_y:
                    move1_pos = ship.position.translate((0,1),board_size)
                elif ship_y < enemy_ship_y:
                    move1_pos = ship.position.translate((0,-1),board_size)

                if move1_pos in df_ship_dispatch.move1_pos:
                    # keep ship in same position to avoid collision with allied ship
#                         print('dispatch - stay in position - 1')
                    df_ship_dispatch.loc[ship_idx,'move1_pos'] = ship.position
#                         print('dispatch - stay in position - 2')
                    ship.next_action = None
                else:                
                    # if no ship conflicts with move1_pos (next position of all ships), we send current ship to intended position
#                         print('dispatch - move - 1')
                    df_ship_dispatch.loc[ship_idx,'move1_pos'] = move1_pos
#                         print('dispatch - move - 2')
                    ship.next_action = move_ship(ship.position,move1_pos,board_size)
                
                if PRINT_MODE:
                    print(f'ESCAPE ENEMY SHIP -- enemy ship distance: {enemy_ship_dist}, enemy ship halite: {enemy_ship_halite}; allied ship halite: {ship.halite}')
    
#     print(df_ship_dispatch)

    if PRINT_MODE:
        print(f'step: {board.step}, ship_count: {len(me.ships)}, shipyards: {len(me.shipyards)}, halite: {me.halite}; shipyard_spawn: {shipyard_spawn}, ship_spawn: {ship_spawn}, total_cost: {ship_spawn*board.configuration.spawn_cost + shipyard_spawn*board.configuration.convert_cost}; board_halite_sum: {board_halite_sum}, halite_to_ship_ratio: {halite_to_ship_ratio}')
    
    return me.next_actions
