extends Node2D

@export var agent : PackedScene
@export var world : TileMapLayer
@export var r_vision : TileMapLayer
@export var c_vision : TileMapLayer

signal next_turn(index: int)
#manages turn order
func _ready() -> void:
	for child : Agent in get_children():
		child.finished_turn.connect(_on_child_fished_turn)
	

func _on_child_fished_turn(index : int):
	#await get_tree().process_frame
	get_child(index).set_deferred("active",false)
	if index == get_child_count() - 1:
		next_turn.emit(0)
		get_child(0).set_deferred("active",true)
	else:
		next_turn.emit(index + 1)
		get_child(index + 1).set_deferred("active",true)

func spawn_agents(runners : int, chasers : int):
	for i in runners:
		print("r")
		var runner : Agent = agent.instantiate()
		runner.character = "RUNNER"
		runner.world = world
		runner.vision = r_vision
		var pos = Vector2i(randi_range(2,world.get_used_rect().size.x-2),randi_range(2,world.get_used_rect().size.y-2))
		runner.global_position = world.local_to_map(pos)
		add_child(runner)
	for i in chasers:
		print("r")
		var chaser : Agent = agent.instantiate()
		chaser.character = "CHASER"
		chaser.world = world
		chaser.vision = r_vision
		var pos = Vector2i(randi_range(2,world.get_used_rect().size.x-2),randi_range(2,world.get_used_rect().size.y-2))
		chaser.global_position = world.local_to_map(pos)
		add_child(chaser)
	get_child(0).active = true
