extends TileMapLayer

var astar = AStarGrid2D.new()

#var start_point = Vector2i()
#var end_point = Vector2i()
#var path = PackedVector2Array()
@export var rect : Rect2i = Rect2i(Vector2i.ZERO, Vector2i(32,32))
@export var wall_chance : float = 0.1
@export var exit_chance : float = 0.1
@export var exit_ammount : int = 1

func _ready():
	generate()
	astar.region = get_used_rect()
	astar.cell_size = tile_set.tile_size
	astar.offset = tile_set.tile_size * 0.5
	astar.diagonal_mode = 1
	astar.update()
	update_astar()


func generate():
	for i in rect.size.x:
		for j in rect.size.y:
			if i == 0 or j == 0 or i == rect.size.x-1 or j == rect.size.y-1:
				set_cell(Vector2i(i,j),0,Global.tiles.BLOCK,0)
				continue
			if randf() <= wall_chance:
				set_cell(Vector2i(i,j),0,Global.tiles.BLOCK,0)
				continue
			if randf() <= exit_chance and exit_ammount > 0:
				exit_ammount -= 1
				set_cell(Vector2i(i,j),0,Global.tiles.EXIT,0)

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
