extends Node2D

@export var world : TileMapLayer

func _ready() -> void:
	Global.runner_won.connect(print.bind("runner won"))
	Global.chaser_won.connect(print.bind("chaser won"))
	#Global.get_state(world, world.get_used_rect())
