extends Label

@export var agents : Node2D

func start():
	agents.next_turn.connect(change_text)
	change_text(0)

func change_text(index : int):
	text = "turn: " + agents.get_child(index).character + " " + str(index)
