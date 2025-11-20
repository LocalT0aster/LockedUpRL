extends Label

func _ready() -> void:
	Global.runner_won.connect(change_text.bind("Runner"))
	Global.chaser_won.connect(change_text.bind("Chaser"))
func change_text(changed_text : String):
	show()
	text = "winner: " + changed_text
