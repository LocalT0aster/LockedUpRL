extends Node2D

@export var world : TileMapLayer
@export var r_vis : TileMapLayer
@export var c_vis : TileMapLayer
@export var agent : PackedScene

var active_agent : Agent
var alt_held : bool

var key_dict := {
	KEY_UP: Vector2i.UP,
	KEY_LEFT: Vector2i.LEFT,
	KEY_RIGHT: Vector2i.RIGHT,
	KEY_DOWN: Vector2i.DOWN
}

signal next_turn(index: int)
#manages turn order

func start():
	spawn_agents()
	for child : Agent in get_children():
		child.finished_turn.connect(_on_child_fished_turn)
	get_child(0).active = true
	active_agent = get_child(0)

func _on_child_fished_turn(index : int):
	#await get_tree().process_frame
	get_child(index).set_deferred("active",false)
	if index == get_child_count() - 1:
		active_agent = get_child(0)
		next_turn.emit(0)
		get_child(0).set_deferred("active",true)
	else:
		active_agent = get_child(index + 1)
		next_turn.emit(index + 1)
		get_child(index + 1).set_deferred("active",true)

func _unhandled_input(event: InputEvent) -> void:
	if Input.is_action_pressed("ui_cancel"):
		_on_child_fished_turn(active_agent.get_index())
		return
	if Input.is_action_pressed("alt"):
		alt_held = true
	else:
		alt_held = false
	if event is InputEventKey:
		if event.pressed and key_dict.has(event.keycode):
			if alt_held and active_agent.character == "CHASER":
				active_agent.place(key_dict[event.keycode])
			else:
				active_agent.go(key_dict[event.keycode])

func spawn_agents():
	for coord in world.get_used_cells_by_id(0,Global.tiles.RUNNER,0):
		var agent_inst = agent.instantiate()
		agent_inst.world = world
		agent_inst.vision =  r_vis
		agent_inst.position = Vector2(coord) * Global.TILE_SIZE + Vector2.ONE * 8
		add_child(agent_inst)
	for coord in world.get_used_cells_by_id(0,Global.tiles.CHASER,0):
		var agent_inst = agent.instantiate()
		agent_inst.world = world
		agent_inst.vision =  c_vis
		agent_inst.character = "CHASER"
		agent_inst.position = Vector2(coord) * Global.TILE_SIZE + Vector2.ONE * 8
		add_child(agent_inst)

func check_runner_win(coords : Vector2i):
	if (
			world.get_cell_atlas_coords(Vector2i(coords)) == Global.tiles.EXIT
		) and (
			active_agent.character == "RUNNER"
		):
			Global.runner_won.emit()

func check_chaser_win():
	for agent : Agent in get_children():
		if (
				world.can_exit(agent.position) 
			): break
	Global.chaser_won.emit()
