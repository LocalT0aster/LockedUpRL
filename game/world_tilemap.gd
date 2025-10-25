extends TileMapLayer

var astar = AStarGrid2D.new()

#var start_point = Vector2i()
#var end_point = Vector2i()
#var path = PackedVector2Array()

func _ready():
	astar.region = get_used_rect()
	astar.cell_size = tile_set.tile_size
	astar.offset = tile_set.tile_size * 0.5
	astar.diagonal_mode = 1
	astar.update()
	update_astar()

func update_astar():
	for i in range(astar.region.position.x, astar.region.end.x):
		for j in range(astar.region.position.y, astar.region.end.y):
			var pos = Vector2i(i, j)
			
			if (
				get_cell_atlas_coords(pos) != Global.tiles.EMPTY
			) and (
				get_cell_atlas_coords(pos) != Global.tiles.EXIT
			):
				astar.set_point_solid(pos, true)
			else:
				astar.set_point_solid(pos, false)
	astar.update()

func can_exit(local_position):
	var map_position = local_to_map(local_position)
	for pos in get_used_cells_by_id(0, Global.tiles.EXIT, 0):
		if astar.get_point_path(map_position, pos) != PackedVector2Array([]):
			return true
	return false
