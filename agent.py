"""
Halite AI Agent - Top 2% of 6,000+ competitors
Two Sigma's Halite Competition (Kaggle, 2020)

Strategy: Multi-step lookahead optimization with attack/escape heuristics
"""

import numpy as np
import pandas as pd
from kaggle_environments.envs.halite.helpers import *

# Configuration
FACTOR_NEAREST_SHIPYARD = 0.25  # Convert to shipyard if farther than this * board_size
ENEMY_DIST_THRESHOLD = 2        # Trigger attack/escape when enemy within this distance
HALITE_THRESHOLD = 500          # Return to shipyard when carrying this much halite
HALITE_TO_SHIP_THRESHOLD = 1400 # Spawn new ships when halite/ship ratio exceeds this


def shipyard_halite_density(board):
    """Calculate halite-to-ship ratio around each shipyard for smart spawning."""
    me = board.current_player
    board_size = board.configuration.size
    box_size = int(FACTOR_NEAREST_SHIPYARD * board_size)
    box_range = range(-box_size, box_size)

    shipyard_list = []
    for shipyard in me.shipyards:
        shipyard_x, shipyard_y = shipyard.position

        # Count ships in area
        ships_in_area = 1
        for ship in me.ships:
            ship_x, ship_y = ship.position
            dist = abs(shipyard_x - ship_x) + abs(shipyard_y - ship_y)
            if dist < box_size:
                ships_in_area += 1

        # Sum halite in area
        halite_in_area = 0
        for i in box_range:
            for j in box_range:
                halite_in_area += board[shipyard.position.translate((i, j), board_size)].halite

        halite_to_ship_ratio = halite_in_area / ships_in_area
        shipyard_list.append({
            'shipyard.id': shipyard.id,
            'shipyard.position': shipyard.position,
            'halite_in_area': round(halite_in_area, 2),
            'ships_in_area': ships_in_area,
            'halite_to_ship_ratio': round(halite_to_ship_ratio, 2)
        })

    df = pd.DataFrame(shipyard_list)
    return df.sort_values('halite_to_ship_ratio', ascending=False)


def ship_dispatcher(board):
    """Orchestrate ship collection and deposit with 4-move lookahead."""
    columns = ['ship.id', 'ship.halite', 'ship.status', 'ship.position',
               'target_ship_x', 'target_ship_y',
               'move1', 'move2', 'move3', 'move4',
               'move1_pos', 'move2_pos', 'move3_pos', 'move4_pos',
               'move1_hal', 'move2_hal', 'move3_hal', 'move4_hal', 'total_hal']
    df_ship_dispatch = pd.DataFrame(columns=columns)
    board_size = board.configuration.size
    me = board.current_player

    moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]  # stay, N, S, W, E
    free_ships = [ship for ship in me.ships if ship.next_action is None]
    ship_conversion_list = []

    # Convert ships to shipyards if too far from base
    if me.halite >= 500:
        for ship in free_ships:
            nearest_dist, nearest_pos = nearest_shipyard(ship, ship_conversion_list, board)
            if nearest_dist > board_size * FACTOR_NEAREST_SHIPYARD:
                ship.next_action = ShipAction.CONVERT
                ship_conversion_list.append(ship.position)

    free_ships = [ship for ship in me.ships if ship.next_action is None]
    shipyard_pos_list = [shipyard.position for shipyard in me.shipyards]

    # Handle deposit runs
    for ship in free_ships:
        if ship.halite > HALITE_THRESHOLD:
            ship_x, ship_y = ship.position
            nearest_dist, nearest_pos = nearest_shipyard(ship, ship_conversion_list, board)
            shipyard_x, shipyard_y = nearest_pos

            # Calculate move toward shipyard
            if ship_x < shipyard_x:
                move1_pos = ship.position.translate((1, 0), board_size)
            elif ship_x > shipyard_x:
                move1_pos = ship.position.translate((-1, 0), board_size)
            elif ship_y < shipyard_y:
                move1_pos = ship.position.translate((0, 1), board_size)
            elif ship_y > shipyard_y:
                move1_pos = ship.position.translate((0, -1), board_size)

            # Avoid collisions
            if move1_pos in df_ship_dispatch.move1_pos.tolist():
                move1_pos = ship.position
                ship.next_action = None
            else:
                ship.next_action = move_ship(ship.position, move1_pos, board_size)

            dispatch_dict = {
                'ship.id': ship.id, 'ship.halite': ship.halite,
                'ship.status': 'DEPOSIT', 'ship.position': ship.position,
                'move1_pos': move1_pos
            }
            df_ship_dispatch.loc[len(df_ship_dispatch)] = dispatch_dict

    free_ships = [ship for ship in me.ships if ship.next_action is None]

    # 4-move lookahead for collection
    for ship in free_ships:
        halite_list = []

        for move1 in moves:
            move1_pos = ship.position.translate(move1, board_size)
            if move1_pos in shipyard_pos_list + df_ship_dispatch.move1_pos.tolist():
                continue

            move1_hal = board[move1_pos].halite * 0.25 if move1 == (0, 0) else 0

            for move2 in moves:
                move2_pos = move1_pos.translate(move2, board_size)
                if move2_pos in shipyard_pos_list + df_ship_dispatch.move2_pos.tolist():
                    continue

                if move2 == (0, 0):
                    move2_hal = move1_hal * 0.75 if move1 == (0, 0) else board[move2_pos].halite * 0.25
                else:
                    move2_hal = 0

                for move3 in moves:
                    move3_pos = move2_pos.translate(move3, board_size)
                    if move3_pos in shipyard_pos_list + df_ship_dispatch.move3_pos.tolist():
                        continue

                    if move3 == (0, 0):
                        move3_hal = move2_hal * 0.75 if move2 == (0, 0) else board[move3_pos].halite * 0.25
                    else:
                        move3_hal = 0

                    for move4 in moves:
                        move4_pos = move3_pos.translate(move4, board_size)
                        if move4_pos in shipyard_pos_list + df_ship_dispatch.move4_pos.tolist():
                            continue

                        if move4 == (0, 0):
                            move4_hal = move3_hal * 0.75 if move3 == (0, 0) else board[move4_pos].halite * 0.25
                        else:
                            move4_hal = 0

                        total_hal = move1_hal + move2_hal + move3_hal + move4_hal
                        halite_list.append({
                            'ship.id': ship.id, 'ship.halite': ship.halite,
                            'ship.status': 'COLLECT', 'ship.position': ship.position,
                            'move1': move1, 'move2': move2, 'move3': move3, 'move4': move4,
                            'move1_pos': move1_pos, 'move2_pos': move2_pos,
                            'move3_pos': move3_pos, 'move4_pos': move4_pos,
                            'move1_hal': move1_hal, 'move2_hal': move2_hal,
                            'move3_hal': move3_hal, 'move4_hal': move4_hal,
                            'total_hal': total_hal
                        })

        df_halite = pd.DataFrame(halite_list)
        df_halite = df_halite.sort_values('total_hal', ascending=False)

        if len(df_halite) > 0:
            best_idx = df_halite.index[0]
            df_ship_dispatch.loc[len(df_ship_dispatch)] = df_halite.loc[best_idx]
            ship.next_action = move_ship(ship.position, df_halite.loc[best_idx, 'move1_pos'], board_size)

    return df_ship_dispatch, ship_conversion_list


def move_ship(start_loc, end_loc, board_size):
    """Calculate direction to move from start to end position."""
    start_x, start_y = divmod(start_loc[0], board_size), divmod(start_loc[1], board_size)
    end_x, end_y = divmod(end_loc[0], board_size), divmod(end_loc[1], board_size)
    if start_y < end_y: return ShipAction.NORTH
    if start_y > end_y: return ShipAction.SOUTH
    if start_x < end_x: return ShipAction.EAST
    if start_x > end_x: return ShipAction.WEST


def nearest_shipyard(ship, ship_conversion_list, board):
    """Find manhattan distance to nearest friendly shipyard."""
    me = board.current_player
    shipyard_positions = [s.position for s in me.shipyards] + ship_conversion_list

    nearest_dist = 999999
    nearest_pos = None

    for shipyard_pos in shipyard_positions:
        ship_x, ship_y = ship.position
        yard_x, yard_y = shipyard_pos
        dist = abs(yard_x - ship_x) + abs(yard_y - ship_y)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_pos = shipyard_pos

    return nearest_dist, nearest_pos


def agent(obs, config):
    """Main agent entry point - called each turn."""
    board_size = config.size
    board = Board(obs, config)
    me = board.current_player

    # Calculate resource metrics
    board_halite_sum = sum(board.observation['halite'])
    halite_to_ship_ratio = board_halite_sum / len(me.ships) if me.ships else 999999

    # Initialize first shipyard
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        me.ships[0].next_action = ShipAction.CONVERT

    # Spawn ships at best shipyard when resources allow
    if (me.halite >= 500 and len(me.shipyards) > 0 and
        halite_to_ship_ratio > HALITE_TO_SHIP_THRESHOLD and
        board.step < board.configuration.episode_steps * 0.75):

        if len(me.shipyards) <= 1:
            me.shipyards[0].next_action = ShipyardAction.SPAWN
        else:
            df_density = shipyard_halite_density(board)
            best_id = df_density.iloc[0]['shipyard.id']
            for shipyard in me.shipyards:
                if shipyard.id == best_id:
                    shipyard.next_action = ShipyardAction.SPAWN
                    break

    # Run collection/deposit dispatcher
    df_ship_dispatch, ship_conversion_list = ship_dispatcher(board)

    # Enemy interaction: attack or escape
    enemy_ships = [s for s in board.ships if board.ships[s].player != board.current_player]
    if enemy_ships:
        for ship in me.ships:
            if ship.next_action == ShipAction.CONVERT:
                continue

            ship_x, ship_y = ship.position

            # Find closest enemy
            enemy_data = []
            for enemy_id in enemy_ships:
                enemy = board.ships[enemy_id]
                ex, ey = enemy.position
                dist = abs(ex - ship_x) + abs(ey - ship_y)
                enemy_data.append({'pos': (ex, ey), 'dist': dist, 'halite': enemy.halite})

            enemy_data.sort(key=lambda x: x['dist'])
            closest = enemy_data[0]

            # Attack if enemy has 2x more halite and is close
            if closest['dist'] <= ENEMY_DIST_THRESHOLD * 2 and closest['halite'] > ship.halite * 2:
                ex, ey = closest['pos']
                if ship_x < ex:
                    move_pos = ship.position.translate((1, 0), board_size)
                elif ship_x > ex:
                    move_pos = ship.position.translate((-1, 0), board_size)
                elif ship_y < ey:
                    move_pos = ship.position.translate((0, 1), board_size)
                else:
                    move_pos = ship.position.translate((0, -1), board_size)

                if move_pos not in df_ship_dispatch.move1_pos.tolist():
                    ship.next_action = move_ship(ship.position, move_pos, board_size)

            # Escape if enemy is close and we have more halite
            elif closest['dist'] <= ENEMY_DIST_THRESHOLD and closest['halite'] < ship.halite * 2:
                ex, ey = closest['pos']
                # Move away from enemy
                if ship_x > ex:
                    move_pos = ship.position.translate((1, 0), board_size)
                elif ship_x < ex:
                    move_pos = ship.position.translate((-1, 0), board_size)
                elif ship_y > ey:
                    move_pos = ship.position.translate((0, 1), board_size)
                else:
                    move_pos = ship.position.translate((0, -1), board_size)

                if move_pos not in df_ship_dispatch.move1_pos.tolist():
                    ship.next_action = move_ship(ship.position, move_pos, board_size)

    return me.next_actions
