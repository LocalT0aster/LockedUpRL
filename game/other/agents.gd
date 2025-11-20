extends Node2D


signal next_turn(index: int)
#manages turn order
func _ready() -> void:
	spawn_agents()
	for child : Agent in get_children():
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
	pass
