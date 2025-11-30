extends Node2D

@export var world : TileMapLayer
@export var r_vis : TileMapLayer
@export var c_vis : TileMapLayer
@export var agent_scene : PackedScene
@export var use_model := true

var active_agent : Agent
var alt_held : bool
var awaiting_action := false
var pending_agent_idx := -1

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
	if use_model:
		_connect_model_signals()
		Model.start()
		_request_model_action(active_agent)

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
	check_chaser_win()
	if active_agent.character == "CHASER":
		Global.get_state(c_vis, c_vis.get_used_rect())
	else:
		Global.get_state(r_vis, r_vis.get_used_rect())
	if use_model:
		_request_model_action(active_agent)

func _unhandled_input(event: InputEvent) -> void:
	if use_model:
		return
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
		var agent_inst = agent_scene.instantiate()
		agent_inst.world = world
		agent_inst.vision =  r_vis
		agent_inst.position = Vector2(coord) * Global.TILE_SIZE + Vector2.ONE * 8
		add_child(agent_inst)
	for coord in world.get_used_cells_by_id(0,Global.tiles.CHASER,0):
		var agent_inst = agent_scene.instantiate()
		agent_inst.world = world
		agent_inst.vision =  c_vis
		agent_inst.character = "CHASER"
		agent_inst.position = Vector2(coord) * Global.TILE_SIZE + Vector2.ONE * 8
		add_child(agent_inst)

#func check_runner_win(_coords : Vector2i):
	#pass
	#now located in agent

func check_chaser_win():
	for agent : Agent in get_children():
		if (
				agent.character == "RUNNER" and world.can_exit(agent.position)
			): return
	Global.chaser_won.emit()


func _connect_model_signals() -> void:
	if Model.action_received.is_connected(_on_model_action):
		return
	Model.action_received.connect(_on_model_action)
	Model.stderr_line.connect(func(line: String) -> void: push_warning("[model stderr] " + line))
	Model.process_exited.connect(func() -> void:
		awaiting_action = false
		pending_agent_idx = -1
		push_warning("Model process exited.")
	)


func _request_model_action(agent: Agent) -> void:
	if not use_model or agent == null:
		return
	if awaiting_action:
		return
	if not Model.is_running():
		Model.start()
		if not Model.is_running():
			push_warning("Model process not running; fallback to manual control.")
			use_model = false
			return
	pending_agent_idx = agent.get_index()
	awaiting_action = true

	var meta := _build_meta_line(agent)
	var rows := _build_rows(agent.character)
	Model.send_data(meta, rows)


func _build_rows(type: String = "") -> PackedStringArray:
	var rows := PackedStringArray()
	var map := world
	match type:
		"CHASER":
			map = c_vis
		"RUNNER":
			map = r_vis
	var state := Global.get_state(map, map.get_used_rect())
	for line in state.strip_edges(true, true).split("\n", false):
		var trimmed := line.strip_edges(true, true)
		if trimmed != "":
			rows.append(trimmed)
	return rows


func _build_meta_line(agent: Agent) -> String:
	var pos := world.local_to_map(agent.position)
	var role := "R"
	if agent.character == "CHASER":
		role = "C" + str(agent.get_index() + 1)
	return "role %s pos %d %d" % [role, pos.x, pos.y]


func _on_model_action(action_line: String) -> void:
	if not awaiting_action:
		return
	var agent: Agent = null
	if pending_agent_idx >= 0 and pending_agent_idx < get_child_count():
		agent = get_child(pending_agent_idx)
	if agent == null:
		awaiting_action = false
		return

	var parts := action_line.strip_edges().split(" ", false)
	if parts.size() < 2:
		awaiting_action = false
		return

	var verb := ""
	var dir := ""
	if parts.size() >= 3 and parts[1].to_lower() == "act":
		verb = parts[2].to_lower()
		if parts.size() >= 4:
			dir = parts[3].to_lower()
	else:
		verb = parts[0].to_lower()
		if parts.size() >= 2:
			dir = parts[1].to_lower()

	awaiting_action = false
	pending_agent_idx = -1
	_apply_action_from_model(agent, verb, dir)


func _apply_action_from_model(agent: Agent, verb: String, dir: String) -> void:
	var dir_map := {
		"up": Vector2i.UP,
		"down": Vector2i.DOWN,
		"left": Vector2i.LEFT,
		"right": Vector2i.RIGHT,
	}

	if verb == "move":
		if dir_map.has(dir):
			agent.go(dir_map[dir])
		else:
			agent.finished_turn.emit(agent.get_index())
		return

	if verb == "build":
		if dir_map.has(dir):
			agent.place(dir_map[dir])
		else:
			agent.finished_turn.emit(agent.get_index())
		return

	# stay or unknown
	agent.finished_turn.emit(agent.get_index())
