extends Node2D

@export var world : TileMapLayer
@export var agents : Node2D
@export var runners : int = 1
@export var chasers : int = 3
@export var vision_size : Vector2i

func _ready() -> void:
	Global.runner_won.connect(print.bind("runner won"))
	Global.chaser_won.connect(print.bind("chaser won"))
	init()
	#Global.get_state(world, world.get_used_rect())

func init():
	agents.spawn_agents(runners, chasers)
