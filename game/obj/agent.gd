extends Node2D

class_name Agent

signal finished_turn


@export var world : TileMapLayer
var cell_size : float
var active = false
var alt_held = false


@export_enum("CHASER", "RUNNER") var character: String = "RUNNER"
@export var vision_size := Vector2i(5,5)

var key_dict := {
	KEY_UP: Vector2i.UP,
	KEY_LEFT: Vector2i.LEFT,
	KEY_RIGHT: Vector2i.RIGHT,
	KEY_DOWN: Vector2i.DOWN
}


func _ready() -> void:
	cell_size = float(world.tile_set.tile_size.x)
	world.set_cell(world.local_to_map(position), 0, Global.tiles[character], 0)
	print(world.local_to_map(position) - vision_size / 2)
	print(Global.get_state(world, Rect2i(world.local_to_map(position) - vision_size / 2,vision_size)))

func _unhandled_input(event: InputEvent) -> void:
	if !active: return
	if Input.is_action_pressed("ui_cancel"):
		finished_turn.emit(get_index())
		return
	if Input.is_action_pressed("alt"):
		alt_held = true
	else:
		alt_held = false
	if event is InputEventKey:
		if event.pressed and key_dict.has(event.keycode):
			if alt_held and character == "CHASER":
				place(key_dict[event.keycode])
			else:
				go(key_dict[event.keycode])

func place(direction : Vector2i):
	var place_position_cells := world.local_to_map(global_position) + direction
	if world.get_cell_atlas_coords(Vector2i(place_position_cells)) == Global.tiles.EMPTY:
		world.set_cell(place_position_cells, 0, Global.tiles.BLOCK, 0)
		world.update_astar()
		finished_turn.emit(get_index())
	check_chaser_win()

func go(direction : Vector2i):
	var new_position_cells := world.local_to_map(global_position) + direction
	if world.get_cell_atlas_coords(Vector2i(new_position_cells)) == Global.tiles.EMPTY:
		world.set_cell(world.local_to_map(position), 0, Global.tiles.EMPTY, 0)
		position += direction * cell_size
		world.set_cell(world.local_to_map(position), 0, Global.tiles[character], 0)
		world.update_astar()
		finished_turn.emit(get_index())
	check_runner_win(new_position_cells)
	check_chaser_win()

func check_runner_win(coords : Vector2i):
	if (
			world.get_cell_atlas_coords(Vector2i(coords)) == Global.tiles.EXIT
		) and (
			character == "RUNNER"
		):
			Global.runner_won.emit()

func check_chaser_win():
	if (
			!world.can_exit(position)
		):
			Global.chaser_won.emit()
