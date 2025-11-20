extends TileMapLayer

func update_vision(world : TileMapLayer, rect: Rect2i):
	for i in rect.size.x:
		for j in rect.size.y:
			var cell_coords = world.get_cell_atlas_coords(Vector2i(i,j) + rect.position)
			set_cell(Vector2i(i,j) + rect.position, 0, cell_coords,0)
