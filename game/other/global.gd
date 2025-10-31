extends Node

@warning_ignore("unused_signal")
signal chaser_won
@warning_ignore("unused_signal")
signal runner_won

var tiles := {
	"EMPTY": Vector2i(-1,-1),
	"BLOCK": Vector2i(0,0),
	"CHASER": Vector2i(0,1),
	"RUNNER": Vector2i(1,1),
	"EXIT": Vector2i(2,0),
}

var state_dict := {
	Vector2i(-1,-1) : 0,#"EMPTY" or floor
	Vector2i(0,0): 1,#"BLOCK"
	Vector2i(0,1): 2,#"CHASER"
	Vector2i(1,1): 3,#"RUNNER"
	Vector2i(2,0): 4,#"EXIT"
}

func get_state(world : TileMapLayer, rect: Rect2i) -> Array[Array]:
	var state: Array[Array] = []
	for i in rect.size.x:
		state.append([]) 
		for j in rect.size.y:
			var a_coords = world.get_cell_atlas_coords(Vector2i(i,j) + rect.position)
			state[i].append(state_dict[a_coords])
	return state
