extends Label

@export var agents : Node2D

func _ready() -> void:
	agents.next_turn.connect(change_text)
	change_text(0)

func change_text(index : int):
	if get_child_count() == 0: return
	text = "turn: " + agents.get_child(index).character + " " + str(index)
