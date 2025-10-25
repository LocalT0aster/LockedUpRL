extends Node2D

@export var world : TileMapLayer

func _ready() -> void:
	Global.runner_won.connect(print.bind("runner won"))
	Global.chaser_won.connect(print.bind("chaser won"))
	get_state()

var state_dict := {
	Vector2i(-1,-1) : 0,#"EMPTY" or floor
	Vector2i(0,0): 1,#"BLOCK"
	Vector2i(0,1): 2,#"CHASER"
	Vector2i(1,1): 3,#"RUNNER"
	Vector2i(2,0): 4,#"EXIT"
}

func get_state() -> Array[Array]:
	var state: Array[Array] = []
	var rect = world.get_used_rect()
	for i in rect.size.x:
		state.append([]) 
		for j in rect.size.y:
			var a_coords = world.get_cell_atlas_coords(Vector2i(i,j) - rect.position)
			state[i].append(state_dict[a_coords])
	return state
