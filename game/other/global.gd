extends Node

@warning_ignore("unused_signal")
signal chaser_won
@warning_ignore("unused_signal")
signal runner_won

const TILE_SIZE = 16.0
var tiles := {
	"UNKNOWN": Vector2i(0,2),
	"EMPTY": Vector2i(-1,-1),
	"FLOOR": Vector2i(2,1),
	"BLOCK": Vector2i(0,0),
	"CHASER": Vector2i(0,1),
	"RUNNER": Vector2i(1,1),
	"EXIT": Vector2i(2,0),
}

var state_dict := {
	Vector2i(0, 2): "unkC",#"UNKNOWN"
	Vector2i(-1,-1) : "_C",#"EMPTY"
	Vector2i(2,1): "_C", #"FLOOR"
	Vector2i(0,0): "oC",#"BLOCK"
	Vector2i(0,1): "acC",#"CHASER"
	Vector2i(1,1): "aeC",#"RUNNER"
	Vector2i(2,0): "eC",#"EXIT"
}

func get_state(world : TileMapLayer, rect: Rect2i) -> String:
	var state: String = ""
	for i in rect.size.x:
		for j in rect.size.y:
			var a_coords = world.get_cell_atlas_coords(Vector2i(i,j) + rect.position)
			state += state_dict[a_coords] + " "
		state += "\n"
	#print(state)
	return state
