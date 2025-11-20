extends Label

@export var agents : Node2D

func _ready() -> void:
	agents.next_turn.connect(change_text)

func change_text(index : int):
	text = "turn: " + agents.get_child(index).character + " " + str(index)
