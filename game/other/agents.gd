extends Node2D

@export var world : TileMapLayer
@export var r_vis : TileMapLayer
@export var c_vis : TileMapLayer
@export var agent : PackedScene

signal next_turn(index: int)
#manages turn order

func start():
	spawn_agents()
	for child : Agent in get_children():
		print(child.position)
		child.finished_turn.connect(_on_child_fished_turn)
	get_child(0).active = true

func _on_child_fished_turn(index : int):
	#await get_tree().process_frame
	get_child(index).set_deferred("active",false)
	if index == get_child_count() - 1:
		next_turn.emit(0)
		get_child(0).set_deferred("active",true)
	else:
		next_turn.emit(index + 1)
		get_child(index + 1).set_deferred("active",true)

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
