extends Node2D

class_name Agent

signal finished_turn


@export var world : TileMapLayer
@export var vision : TileMapLayer
var active = false #not in use atm

@export_enum("CHASER", "RUNNER") var character: String = "RUNNER"
@export var vision_size := Vector2i(5,5)

func _ready() -> void:
	#world.set_cell(world.local_to_map(position), 0, Global.tiles[character], 0)
	vision.update_vision(world, Rect2i(world.local_to_map(position) - vision_size / 2,vision_size))
	#print(world.local_to_map(position) - vision_size / 2)
	#print(Global.get_state(vision, world.get_used_rect()))

func place(direction : Vector2i):
	var place_position_cells := world.local_to_map(global_position) + direction
	if world.get_cell_atlas_coords(Vector2i(place_position_cells)) == Global.tiles.FLOOR:
		world.set_cell(place_position_cells, 0, Global.tiles.BLOCK, 0)
		world.update_astar()
		vision.update_vision(world, Rect2i(world.local_to_map(position) - vision_size / 2,vision_size))
		finished_turn.emit(get_index())

func go(direction : Vector2i):
	var new_position_cells := world.local_to_map(global_position) + direction
	if world.get_cell_atlas_coords(Vector2i(new_position_cells)) == Global.tiles.FLOOR:
		world.set_cell(world.local_to_map(position), 0, Global.tiles.FLOOR, 0)
		position += direction * Global.TILE_SIZE
		world.set_cell(world.local_to_map(position), 0, Global.tiles[character], 0)
		world.update_astar()
		vision.update_vision(world, Rect2i(world.local_to_map(position) - vision_size / 2,vision_size))
		finished_turn.emit(get_index())
