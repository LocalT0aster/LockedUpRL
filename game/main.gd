extends Node2D

@export var world : TileMapLayer

func _ready() -> void:
	init()
	Global.runner_won.connect(print.bind("runner won"))
	Global.chaser_won.connect(print.bind("chaser won"))
	print("world")
	Global.get_state(world, world.get_used_rect())

func init():
	$world_tilemap.start()
	$agents.start()
	$CanvasLayer/Label.start()
